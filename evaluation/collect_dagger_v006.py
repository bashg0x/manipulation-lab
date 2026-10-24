#!/usr/bin/env python3
"""Collect 27-second ACT->V005-teacher trajectories with expert labels at ALL steps.

Diagnostic dataset (not ACT-only success). ACT controls the initial interval;
V005 teacher controls the remainder. Teacher IK labels, RGB and observations
are collected at 50 Hz, including during the ACT-controlled interval.

Run from repository root with PYTHONPATH=$PWD; requires existing local modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from act_policy_adapter import ActPolicyAdapter
from rollout_act import ROOT, DEFAULT_MODEL, resolve_arm, resolve_gripper_joint, reset, clip_ctrl, ARM_JOINT_NAMES
from mujoco_generator.controllers.pose_ik_controller import PoseIKController
from mujoco_generator.controllers.pick_place_controller_v005 import PickPlaceControllerV005

CONTROL_DT = 0.02


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--switch-seconds", type=float, required=True)
    p.add_argument("--max-seconds", type=float, default=27.0)
    p.add_argument("--cube-offset-mm", nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument("--replan-every", type=int, default=0)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    p.add_argument("--target-xy", nargs=2, type=float, default=[0.662, 0.234])
    p.add_argument("--table-z", type=float, default=0.1522)
    p.add_argument("--cube-halfheight", type=float, default=0.025)
    p.add_argument("--image-every", type=int, default=1, help="Save every N frames. Use 1 for training.")
    args = p.parse_args()

    if not 0 < args.switch_seconds < args.max_seconds:
        p.error("Require 0 < switch-seconds < max-seconds")
    if args.image_every != 1:
        p.error("Use --image-every 1: training requires one image per control frame")
    for key, value in (("switch-seconds", args.switch_seconds), ("max-seconds", args.max_seconds)):
        if not np.isclose(value / CONTROL_DT, round(value / CONTROL_DT)):
            p.error(f"--{key} must be a multiple of 0.02")

    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite nonempty output: {output}")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    arm_qpos, arm_act, grip_act = resolve_arm(model)
    grip_qpos, gripper_name = resolve_gripper_joint(model, None)
    arm_dof = np.array([int(model.jnt_dofadr[model.joint(name).id]) for name in ARM_JOINT_NAMES])
    cube_body = reset(model, data, arm_qpos, arm_act, grip_act, args.cube_offset_mm)

    tcp_id = model.site("gripper_pinch").id
    target_id = model.site("place_target").id
    home_position = data.site_xpos[tcp_id].copy()
    home_rotation = data.site_xmat[tcp_id].copy().reshape(3, 3)
    cube_initial = data.xpos[cube_body].copy()
    place_target = data.site_xpos[target_id].copy()
    geom = model.geom("worktable")
    table_top = float(geom.pos[2] + geom.size[2])

    teacher = PickPlaceControllerV005(
        home_position=home_position,
        cube_position=cube_initial,
        place_target=place_target,
        table_top=table_top,
    )
    ik = PoseIKController(model)
    policy = ActPolicyAdapter(ROOT, args.checkpoint, args.device)
    replan_n = args.replan_every or policy.chunk_size
    if not 1 <= replan_n <= policy.chunk_size:
        p.error("replan-every must be <= ACT chunk size")
    if int(policy.skip) < 1:
        raise ValueError("ACT skip must be positive")
    physics_steps = round(CONTROL_DT / model.opt.timestep)
    if not np.isclose(physics_steps * model.opt.timestep, CONTROL_DT):
        raise ValueError("MuJoCo timestep must divide 0.02")

    output.mkdir(parents=True, exist_ok=True)
    images = output / "images" / "front"
    images.mkdir(parents=True, exist_ok=True)
    renderer = mujoco.Renderer(model, height=240, width=320)

    keys = (
        "timestamp", "joint_position", "joint_velocity", "gripper_position",
        "action", "act_command", "tcp_position", "cube_position",
        "phase", "control_mode", "inference_step",
    )
    history = {k: [] for k in keys}
    chunk = None
    chunk_index = 0
    hold_step = 0
    current_action = None
    configured = False
    num_inferences = 0
    max_clearance = -np.inf
    num_steps = round(args.max_seconds / CONTROL_DT)
    switch_step = round(args.switch_seconds / CONTROL_DT)
    applied_ids = np.r_[arm_act, grip_act]

    print("=== DAgger V006 full-episode collection ===", flush=True)
    print(f"ACT 0-{args.switch_seconds}s; teacher {args.switch_seconds}-{args.max_seconds}s", flush=True)
    print("Expert labels computed at every 50Hz frame; ACT labels do not control simulation", flush=True)
    try:
        for step in range(num_steps):
            t = step * CONTROL_DT
            obs_time = float(data.time)
            state = np.r_[data.qpos[arm_qpos], data.qpos[grip_qpos]].astype(np.float64)
            joint_q = data.qpos[arm_qpos].copy()
            joint_dq = data.qvel[arm_dof].copy()
            grip_angle = float(data.qpos[grip_qpos])
            tcp = data.site_xpos[tcp_id].copy()
            cube = data.xpos[cube_body].copy()

            renderer.update_scene(data, camera="front")
            rgb = renderer.render().copy()

            # Keep V005 task target generation and online IK identical to teacher collector.
            if not configured and t >= teacher.T_TRANSFER_HOLD:
                teacher.configure_place(
                    tcp_position=data.site_xpos[tcp_id].copy(),
                    cube_position=data.xpos[cube_body].copy(),
                )
                configured = True
                print(f"Teacher configure_place at t={t:.2f}s", flush=True)
            task = teacher.get_action(t)
            expert_joints = np.asarray(ik.solve(
                target_position=task.target_position,
                target_rotation=home_rotation,
                current_qpos=data.qpos.copy(),
            ), dtype=np.float64).reshape(-1)
            if expert_joints.shape != (6,):
                raise ValueError(f"IK returned shape {expert_joints.shape}, expected (6,)")
            expert = np.r_[expert_joints, float(task.gripper_command)]
            if not np.all(np.isfinite(expert)):
                raise ValueError(f"Non-finite expert action at step {step}")

            inference = False
            if step < switch_step:
                if hold_step == 0:
                    if chunk is None or chunk_index >= replan_n:
                        chunk = policy.predict_chunk(state, rgb)
                        chunk_index = 0
                        num_inferences += 1
                        inference = True
                    current_action = np.asarray(chunk[chunk_index]).copy()
                    chunk_index += 1
                desired = current_action
                hold_step = (hold_step + 1) % policy.skip
                control_mode = 0
            else:
                if step == switch_step:
                    print(f"Switching to teacher at t={t:.2f}s", flush=True)
                desired = expert
                control_mode = 1

            for j, actuator in enumerate(arm_act):
                data.ctrl[int(actuator)] = clip_ctrl(model, int(actuator), float(desired[j]))
            data.ctrl[int(grip_act)] = clip_ctrl(
                model, int(grip_act), float(np.clip(desired[6], 0.0, 255.0))
            )

            history["timestamp"].append(obs_time)
            history["joint_position"].append(joint_q)
            history["joint_velocity"].append(joint_dq)
            history["gripper_position"].append(grip_angle)
            history["action"].append(expert.copy())
            history["act_command"].append(data.ctrl[applied_ids].copy())
            history["tcp_position"].append(tcp)
            history["cube_position"].append(cube)
            history["phase"].append(str(task.phase))
            history["control_mode"].append(control_mode)
            history["inference_step"].append(inference)
            Image.fromarray(rgb).save(images / f"{step:06d}.png")

            max_clearance = max(
                max_clearance,
                float(cube[2] - args.cube_halfheight - args.table_z),
            )
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
            if (step + 1) % 250 == 0:
                print(f"  {step + 1}/{num_steps} frames", flush=True)
    finally:
        renderer.close()

    arrays = {k: np.asarray(v) for k, v in history.items()}
    if len(arrays["timestamp"]) != num_steps:
        raise RuntimeError("Unexpected number of frames")
    if not np.all(np.diff(arrays["timestamp"]) > 0):
        raise RuntimeError("Non-increasing timestamp")
    if not np.all(np.isfinite(arrays["action"])):
        raise RuntimeError("Non-finite labels")
    if len(list(images.glob("*.png"))) != num_steps:
        raise RuntimeError("RGB image count mismatch")

    cube_final = data.xpos[cube_body].copy()
    xy_error = float(np.linalg.norm(cube_final[:2] - args.target_xy))
    height_error = float(abs(cube_final[2] - args.cube_halfheight - args.table_z))
    geometric_success = bool(max_clearance > 0.05 and xy_error < 0.03 and height_error < 0.015)
    arm_rmse = np.sqrt(np.mean(
        (arrays["action"][:switch_step, :6] - arrays["act_command"][:switch_step, :6]) ** 2,
        axis=1,
    ))
    metadata = {
        "type": "DAGGER_V006_EXPERT_LABELS_ACT_TO_TEACHER",
        "label_semantics": "action=teacher IK label in every frame; act_command=applied control",
        "control_mode": {"0": "ACT", "1": "V005 teacher"},
        "checkpoint": str(args.checkpoint.resolve()),
        "model": str(args.model.resolve()),
        "switch_seconds": args.switch_seconds,
        "duration_seconds": args.max_seconds,
        "control_dt": CONTROL_DT,
        "frames": num_steps,
        "cube_offset_mm": args.cube_offset_mm,
        "gripper_joint": gripper_name,
        "act_chunk_size": policy.chunk_size,
        "act_skip": policy.skip,
        "act_replan_every": replan_n,
        "act_inference_calls": num_inferences,
        "teacher_place_configured": configured,
        "xy_error_mm": xy_error * 1000,
        "height_error_mm": height_error * 1000,
        "max_clearance_mm": max_clearance * 1000,
        "provisional_geometric_success": geometric_success,
        "act_interval_expert_command_rmse_mean_rad": float(arm_rmse.mean()),
        "act_interval_expert_command_rmse_max_rad": float(arm_rmse.max()),
        "warning": (
            "Do NOT interpret this as ACT-only success. Off-trajectory labels use a "
            "time-conditioned teacher and must be validated before training."
        ),
    }
    np.savez_compressed(output / "episode.npz", **arrays)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    print("Saved:", output, flush=True)


if __name__ == "__main__":
    main()
