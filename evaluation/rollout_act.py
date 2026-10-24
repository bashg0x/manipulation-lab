"""Roll out a trained ACT checkpoint in the independent MuJoCo Pick & Place model.

Example:
  python evaluation/rollout_act.py --max-seconds 29 --cube-offset-mm 0 0

Unlike teacher-data collection, this script NEVER calls the teacher controller.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from act_policy_adapter import ActPolicyAdapter

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = ROOT / "mujoco_generator/environments/pick_and_place_camera.xml"
DEFAULT_CKPT = ROOT / "training/checkpoints/act_smoke_test/policy_best.ckpt"
DEFAULT_OUTPUT = None
ARM_JOINT_NAMES = (
    "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
    "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
)
HOME_QPOS = np.array([-np.pi / 2, -np.pi / 2, np.pi / 2,
                       -np.pi / 2, -np.pi / 2, 0.0])


def args_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT)
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Unique output directory; default uses checkpoint experiment and replan count")
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--max-seconds", type=float, default=29.0)
    p.add_argument("--cube-offset-mm", type=float, nargs=2, default=[0., 0.])
    p.add_argument("--target-xy", type=float, nargs=2, default=[0.662, 0.234])
    p.add_argument("--table-z", type=float, default=0.1522)
    p.add_argument("--cube-halfheight", type=float, default=0.025)
    p.add_argument("--gripper-joint", type=str, default=None,
                   help="Observed gripper joint. Default: auto-detect driver joint")
    p.add_argument("--replan-every", type=int, default=0,
                   help="Replan after N model actions; 0 means full action chunk")
    p.add_argument("--snapshot-every", type=float, default=2.0,
                   help="Seconds between saved front camera PNGs (NOT frames); 0 disables")
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--height", type=int, default=240)
    return p.parse_args()


def name_list(model, objtype, count):
    return [mujoco.mj_id2name(model, objtype, i) or f"unnamed_{i}"
            for i in range(count)]


def resolve_gripper_joint(model, chosen):
    names = name_list(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    if chosen is None:
        favored = ["gripper_right_driver_joint", "gripper_left_driver_joint",
                   "left_driver_joint", "finger_joint", "gripper_joint"]
        for name in favored:
            if name in names:
                chosen = name
                break
    if chosen not in names:
        raise ValueError(
            f"Cannot resolve measured gripper joint {chosen!r}. "
            f"Available joint names: {names}. Pass --gripper-joint NAME"
        )
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, chosen)
    return int(model.jnt_qposadr[jid]), chosen


def resolve_arm(model):
    names = name_list(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt)
    missing = [x for x in ARM_JOINT_NAMES if x not in names]
    if missing:
        raise ValueError(f"Missing UR5e joints: {missing}; available: {names}")
    qpos_ids = [int(model.jnt_qposadr[mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, n)]) for n in ARM_JOINT_NAMES]
    # Map arm position actuator by its joint transmission; never assume order.
    actuator_ids = []
    for joint_name in ARM_JOINT_NAMES:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        matches = [i for i in range(model.nu)
                   if int(model.actuator_trntype[i]) == int(mujoco.mjtTrn.mjTRN_JOINT)
                   and int(model.actuator_trnid[i, 0]) == jid]
        if len(matches) != 1:
            raise ValueError(f"Expected one joint actuator for {joint_name}: got {matches}")
        actuator_ids.append(matches[0])
    gripper_act = [i for i in range(model.nu) if i not in actuator_ids]
    if len(gripper_act) != 1:
        raise ValueError(f"Expected 1 gripper actuator, got IDs {gripper_act}")
    return np.array(qpos_ids, int), np.array(actuator_ids, int), gripper_act[0]


def clip_ctrl(model, actuator_id, value):
    if bool(model.actuator_ctrllimited[actuator_id]):
        low, high = model.actuator_ctrlrange[actuator_id]
        return float(np.clip(value, low, high))
    return float(value)


def reset(model, data, arm_qpos_ids, arm_acts, gripper_act, cube_offset_mm):
    mujoco.mj_resetData(model, data)
    data.qpos[arm_qpos_ids] = HOME_QPOS
    for a, q in zip(arm_acts, HOME_QPOS):
        data.ctrl[a] = clip_ctrl(model, a, q)
    data.ctrl[gripper_act] = clip_ctrl(model, gripper_act, 0.0)

    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cube_freejoint")
    if jid < 0:
        raise ValueError("MuJoCo joint 'cube_freejoint' not found")
    if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise ValueError("'cube_freejoint' is not a free joint")
    cube_body_id = int(model.jnt_bodyid[jid])
    cube_qpos = int(model.jnt_qposadr[jid])
    data.qpos[cube_qpos:cube_qpos + 2] += np.asarray(cube_offset_mm) / 1000.0
    mujoco.mj_forward(model, data)
    return cube_body_id


def main():
    args = args_parser()
    if args.max_seconds <= 0 or args.replan_every < 0 or args.snapshot_every < 0:
        raise ValueError("Invalid duration, replan interval or snapshot interval")
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    arm_qpos_ids, arm_act_ids, gripper_act_id = resolve_arm(model)
    gripper_qpos_id, gripper_joint_name = resolve_gripper_joint(model, args.gripper_joint)
    cube_body_id = reset(model, data, arm_qpos_ids, arm_act_ids,
                         gripper_act_id, args.cube_offset_mm)
    adapter = ActPolicyAdapter(ROOT, args.checkpoint, args.device)

    control_dt = 0.02
    steps_per_control = round(control_dt / model.opt.timestep)
    if not np.isclose(steps_per_control * model.opt.timestep, control_dt):
        raise ValueError(f"MuJoCo dt {model.opt.timestep} does not divide 0.02 sec")
    num_steps = round(args.max_seconds / control_dt)
    replan_n = args.replan_every or adapter.chunk_size
    if replan_n > adapter.chunk_size:
        raise ValueError("--replan-every cannot exceed ACT chunk size")
    # Time alignment: each predicted command is held for skip*0.02 seconds.
    command_hold = adapter.skip
    chunk = None
    replan_index = 0
    steps_since_action = 0
    current_action = None
    inference_happened = False
    max_clearance = -np.inf
    history = {
        "time": [], "state": [], "predicted_action": [],
        "command": [], "cube_position": [], "inference_step": [],
    }
    if args.output_dir is None:
        run_name = f"{args.checkpoint.parent.name}_replan{replan_n}"
        args.output_dir = ROOT / "evaluation" / "results" / run_name
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory is not empty: {args.output_dir}. "
            "Use --output-dir to choose a new directory."
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    next_snapshot = 0.0
    snapshot_dir = args.output_dir / "snapshots"
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    print("=== ACT MuJoCo Rollout ===")
    print("Model:", args.model)
    print("Checkpoint:", args.checkpoint)
    print("Gripper observed joint:", gripper_joint_name)
    print("Arm actuator IDs:", arm_act_ids.tolist(), "Gripper actuator ID:", gripper_act_id)
    print("ACT chunk:", adapter.chunk_size, "skip:", adapter.skip,
          "replan every model actions:", replan_n)
    try:
        for step in range(num_steps):
            t = step * control_dt
            inference_happened = False
            state = np.concatenate([data.qpos[arm_qpos_ids],
                                    [data.qpos[gripper_qpos_id]]]).astype(np.float64)
            if steps_since_action == 0:
                if chunk is None or replan_index >= replan_n:
                    renderer.update_scene(data, camera="front")
                    rgb = renderer.render().copy()
                    chunk = adapter.predict_chunk(state, rgb)
                    replan_index = 0
                    inference_happened = True
                current_action = chunk[replan_index].copy()
                replan_index += 1
                # Physical constraints, but don't conceal bad predictions in saved trajectory
            for i, actuator in enumerate(arm_act_ids):
                data.ctrl[actuator] = clip_ctrl(model, actuator, current_action[i])
            gripper_command = np.clip(current_action[6], 0.0, 255.0)
            data.ctrl[gripper_act_id] = clip_ctrl(model, gripper_act_id, gripper_command)

            cube = data.xpos[cube_body_id].copy()
            bottom = cube[2] - args.cube_halfheight
            max_clearance = max(max_clearance, bottom - args.table_z)
            history["time"].append(t)
            history["state"].append(state)
            history["predicted_action"].append(current_action.copy())
            history["command"].append(data.ctrl[np.r_[arm_act_ids, gripper_act_id]].copy())
            history["inference_step"].append(inference_happened)
            history["cube_position"].append(cube)

            if args.snapshot_every > 0 and t + 1e-9 >= next_snapshot:
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                renderer.update_scene(data, camera="front")
                Image.fromarray(renderer.render().copy()).save(
                    snapshot_dir / f"frame_{step:06d}.png")
                next_snapshot += args.snapshot_every

            for _ in range(steps_per_control):
                mujoco.mj_step(model, data)
            steps_since_action = (steps_since_action + 1) % command_hold
            if (step + 1) % 250 == 0:
                print(f"  {step + 1}/{num_steps} steps")

        cube = data.xpos[cube_body_id].copy()
        xy_error = float(np.linalg.norm(cube[:2] - np.asarray(args.target_xy)))
        height_error = float(abs((cube[2] - args.cube_halfheight) - args.table_z))
        # Same geometric thresholds as the demonstration collector. This is
        # a provisional geometric test, NOT an independent task success label.
        success = bool(max_clearance > 0.05 and xy_error < 0.03 and height_error < 0.015)
        result = {
            "checkpoint": str(args.checkpoint.resolve()),
            "model": str(args.model.resolve()),
            "device": args.device,
            "steps": num_steps,
            "duration_s": num_steps * control_dt,
            "cube_offset_mm": args.cube_offset_mm,
            "replan_every": replan_n,
            "skip": adapter.skip,
            "cube_final_position_m": cube.tolist(),
            "xy_error_mm": xy_error * 1000.0,
            "height_error_mm": height_error * 1000.0,
            "max_clearance_mm": max_clearance * 1000.0,
            "provisional_success": success,
            "gripper_predicted_min_max": [
                float(np.min(np.asarray(history["predicted_action"])[:, 6])),
                float(np.max(np.asarray(history["predicted_action"])[:, 6])),
            ],
            "gripper_applied_min_max": [
                float(np.min(np.asarray(history["command"])[:, 6])),
                float(np.max(np.asarray(history["command"])[:, 6])),
            ],
            "gripper_negative_prediction_fraction": float(
                np.mean(np.asarray(history["predicted_action"])[:, 6] < 0)
            ),
            "num_inference_calls": int(np.sum(history["inference_step"])),
        }
        (args.output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        np.savez_compressed(args.output_dir / "rollout.npz",
                            **{k: np.asarray(v) for k, v in history.items()})
        print("=== Rollout Result ===")
        print(json.dumps(result, ensure_ascii=False, indent=2))
        print("Saved:", args.output_dir)
    finally:
        renderer.close()


if __name__ == "__main__":
    main()
