#!/usr/bin/env python3
"""Record MuJoCo Pick & Place in teacher, ACT-only, or teacher->ACT mode.

Save synchronized 50Hz telemetry and 25fps camera PNGs. This does NOT train or
modify a checkpoint. Requires the existing project's evaluation/rollout_act.py,
act_policy_adapter.py, and V005 teacher/controller modules.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from act_policy_adapter import ActPolicyAdapter
from rollout_act import ROOT, DEFAULT_MODEL, clip_ctrl, reset, resolve_arm, resolve_gripper_joint
from mujoco_generator.controllers.pose_ik_controller import PoseIKController
from mujoco_generator.controllers.pick_place_controller_v005 import PickPlaceControllerV005

DT = 0.02


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode', choices=['act', 'teacher', 'assisted'], required=True)
    p.add_argument('--checkpoint', type=Path, default=ROOT / 'training/checkpoints/act_experiment_006b/policy_best.ckpt')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--model', type=Path, default=DEFAULT_MODEL)
    p.add_argument('--switch-seconds', type=float, default=10.0, help='Teacher to ACT switch, assisted mode only')
    p.add_argument('--max-seconds', type=float, default=27.0)
    p.add_argument('--cube-offset-mm', nargs=2, type=float, default=[0.0, 0.0])
    p.add_argument('--replan-every', type=int, default=1, help='ACT action steps before replanning (1..25)')
    p.add_argument('--fps', type=int, default=25, help='Camera output fps: 25, 10, 5 or 0 (no images)')
    p.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    p.add_argument('--target-xy', nargs=2, type=float, default=[0.662, 0.234])
    p.add_argument('--table-z', type=float, default=0.1522)
    p.add_argument('--cube-halfheight', type=float, default=0.025)
    args = p.parse_args()

    if args.max_seconds <= 0 or not np.isclose(args.max_seconds / DT, round(args.max_seconds / DT)):
        p.error('max-seconds must be a positive multiple of 0.02')
    if args.mode == 'assisted' and (not 0 < args.switch_seconds < args.max_seconds or not np.isclose(args.switch_seconds / DT, round(args.switch_seconds / DT))):
        p.error('assisted switch-seconds must be a multiple of 0.02 and strictly within the run')
    if args.fps not in (0, 5, 10, 25):
        p.error('fps must be one of 0, 5, 10, 25 (exact divisors of 50)')
    out = args.output_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'Output is not empty: {out}')

    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    arm_qpos, arm_act, grip_act = resolve_arm(model)
    grip_qpos, grip_joint_name = resolve_gripper_joint(model, None)
    cube_body = reset(model, data, arm_qpos, arm_act, grip_act, args.cube_offset_mm)
    tcp_site = model.site('gripper_pinch').id
    target_site = model.site('place_target').id
    initial_tcp = data.site_xpos[tcp_site].copy()
    home_rotation = data.site_xmat[tcp_site].copy().reshape(3, 3)
    cube_initial = data.xpos[cube_body].copy()
    target_position = data.site_xpos[target_site].copy()
    table_geom = model.geom('worktable')
    table_top = float(table_geom.pos[2] + table_geom.size[2])

    teacher = PickPlaceControllerV005(home_position=initial_tcp, cube_position=cube_initial,
                                     place_target=target_position, table_top=table_top)
    ik = PoseIKController(model)
    policy = None
    if args.mode in ('act', 'assisted'):
        policy = ActPolicyAdapter(ROOT, args.checkpoint, args.device)
        if not 1 <= args.replan_every <= policy.chunk_size:
            p.error('replan-every must be between 1 and ACT chunk_size')

    physics_steps = round(DT / model.opt.timestep)
    if not np.isclose(physics_steps * model.opt.timestep, DT):
        raise ValueError('MuJoCo timestep must divide 0.02s')
    frame_stride = (50 // args.fps) if args.fps else None
    num_steps = round(args.max_seconds / DT)
    switch_step = round(args.switch_seconds / DT)
    out.mkdir(parents=True, exist_ok=True)
    frames = out / 'frames'
    if frame_stride:
        frames.mkdir()
    history = {k: [] for k in ('time', 'state', 'command', 'expert_action',
                                'predicted_action', 'cube_position', 'tcp_position',
                                'control_mode', 'inference_step', 'phase')}
    renderer = mujoco.Renderer(model, width=320, height=240)
    placed_config = False
    current_chunk = None
    chunk_index = 0
    hold = 0
    current_action = None
    calls = 0
    clearance_max = -np.inf
    actuator_ids = np.r_[arm_act, grip_act]
    print(f'=== VIDEO RECORD: {args.mode}, t={args.max_seconds}s, offset={args.cube_offset_mm}mm ===', flush=True)
    try:
        for step in range(num_steps):
            t = step * DT
            state = np.r_[data.qpos[arm_qpos], data.qpos[grip_qpos]].astype(np.float64)
            cube = data.xpos[cube_body].copy()
            tcp = data.site_xpos[tcp_site].copy()
            assisted_teacher = args.mode == 'assisted' and step < switch_step
            teacher_control = args.mode == 'teacher' or assisted_teacher
            inference = False
            predicted = np.full(7, np.nan)

            # Expert label for diagnostics at all times, regardless of controlling mode.
            if not placed_config and t >= teacher.T_TRANSFER_HOLD:
                teacher.configure_place(tcp_position=tcp.copy(), cube_position=cube.copy())
                placed_config = True
            task = teacher.get_action(t)
            q_expert = np.asarray(ik.solve(target_position=task.target_position,
                                           target_rotation=home_rotation,
                                           current_qpos=data.qpos.copy()), dtype=np.float64)
            if q_expert.shape != (6,):
                raise ValueError(f'IK unexpected shape {q_expert.shape}')
            expert = np.r_[q_expert, float(task.gripper_command)]
            if teacher_control:
                desired = expert
            else:
                if args.mode == 'assisted' and step == switch_step:
                    print(f'  SWITCH teacher -> ACT at {t:.2f}s (simulation state preserved)', flush=True)
                    current_chunk, chunk_index, hold = None, 0, 0
                if hold == 0:
                    if current_chunk is None or chunk_index >= args.replan_every:
                        renderer.update_scene(data, camera='front')
                        rgb = renderer.render().copy()
                        current_chunk = policy.predict_chunk(state, rgb)
                        chunk_index = 0
                        inference = True
                        calls += 1
                    current_action = current_chunk[chunk_index].copy()
                    chunk_index += 1
                desired = current_action
                predicted = np.asarray(desired).copy()
                hold = (hold + 1) % policy.skip

            for j, actuator in enumerate(arm_act):
                data.ctrl[int(actuator)] = clip_ctrl(model, int(actuator), float(desired[j]))
            grip_command = float(np.clip(desired[6], 0, 255))
            data.ctrl[int(grip_act)] = clip_ctrl(model, int(grip_act), grip_command)
            history['time'].append(t)
            history['state'].append(state)
            history['command'].append(data.ctrl[actuator_ids].copy())
            history['expert_action'].append(expert)
            history['predicted_action'].append(predicted)
            history['cube_position'].append(cube)
            history['tcp_position'].append(tcp)
            history['control_mode'].append(1 if teacher_control else 0)
            history['inference_step'].append(inference)
            history['phase'].append(str(task.phase))
            clearance_max = max(clearance_max, float(cube[2] - args.cube_halfheight - args.table_z))

            if frame_stride and step % frame_stride == 0:
                renderer.update_scene(data, camera='front')
                Image.fromarray(renderer.render().copy()).save(frames / f'frame_{step:06d}.png')
            for _ in range(physics_steps):
                mujoco.mj_step(model, data)
            if (step + 1) % 250 == 0:
                print(f'  {step+1}/{num_steps} simulation steps', flush=True)
    finally:
        renderer.close()

    cube_final = data.xpos[cube_body].copy()
    xy_error = float(np.linalg.norm(cube_final[:2] - args.target_xy))
    height_error = float(abs((cube_final[2] - args.cube_halfheight) - args.table_z))
    success = bool(clearance_max > 0.05 and xy_error < 0.03 and height_error < 0.015)
    arrays = {k: np.asarray(v) for k, v in history.items()}
    assert len(arrays['time']) == num_steps
    np.savez_compressed(out / 'rollout.npz', **arrays)
    result = dict(mode=args.mode, checkpoint=str(args.checkpoint) if policy else None,
                  switch_seconds=args.switch_seconds if args.mode == 'assisted' else None,
                  duration_seconds=args.max_seconds, cube_offset_mm=args.cube_offset_mm,
                  replan_every=args.replan_every if policy else None,
                  fps=args.fps, frame_count=len(list(frames.glob('*.png'))) if frame_stride else 0,
                  inference_calls=calls, observed_gripper_joint=grip_joint_name,
                  xy_error_mm=xy_error*1000, max_clearance_mm=clearance_max*1000,
                  height_error_mm=height_error*1000, provisional_geometric_success=success,
                  control_mode_encoding={'0':'ACT','1':'V005 teacher'},
                  note='Teacher-assisted results must never be reported as ACT-only success.')
    (out / 'result.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    print('Saved:', out, flush=True)


if __name__ == '__main__':
    main()
