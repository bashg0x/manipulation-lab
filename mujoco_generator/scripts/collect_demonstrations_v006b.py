
import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from mujoco_generator.controllers.pose_ik_controller import (
    ARM_JOINTS,
    PoseIKController,
)
from mujoco_generator.controllers.pick_place_controller_v005 import (
    PickPlaceControllerV005,
)
from mujoco_generator.collectors.episode_collector import (
    EpisodeCollector,
)


MODEL_PATH = (
    ROOT
    / "mujoco_generator"
    / "environments"
    / "pick_and_place_camera.xml"
)

OUTPUT_ROOT = ROOT / "data" / "raw_v006b"

HOME_Q = np.array([
    -1.5708,
    -1.5708,
     1.5708,
    -1.5708,
    -1.5708,
     0.0,
])

ARM_ACTUATORS = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow",
    "wrist_1",
    "wrist_2",
    "wrist_3",
)

GRIPPER_ACTUATOR = "gripper_fingers_actuator"
GRIPPER_JOINT = "gripper_right_driver_joint"

CAMERA_NAME = "front"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

CONTROL_DT = 0.02


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect randomized MuJoCo "
            "Pick & Place demonstrations"
        )
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="Number of episodes to collect",
    )

    parser.add_argument(
        "--range-mm",
        type=float,
        default=20.0,
        help="Cube XY randomization range in mm",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    parser.add_argument(
        "--start-id",
        type=int,
        default=2,
        help="Starting episode number",
    )


    parser.add_argument(
        "--perturb-mode",
        choices=["none", "single", "multi"],
        default="none",
    )
    parser.add_argument("--perturb-rad", type=float, default=0.01)
    parser.add_argument("--perturb-time", type=float, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
    )
    return parser.parse_args()


class DemonstrationRunner:
    """MuJoCoで教師エピソードを生成する。"""

    def __init__(self):
        self.model = mujoco.MjModel.from_xml_path(
            str(MODEL_PATH)
        )
        self.data = mujoco.MjData(self.model)

        self.ik = PoseIKController(self.model)

        self.arm_qpos_ids = np.array([
            self.model.jnt_qposadr[
                self.model.joint(name).id
            ]
            for name in ARM_JOINTS
        ])

        self.arm_qvel_ids = np.array([
            self.model.jnt_dofadr[
                self.model.joint(name).id
            ]
            for name in ARM_JOINTS
        ])

        self.arm_actuator_ids = np.array([
            self.model.actuator(name).id
            for name in ARM_ACTUATORS
        ])

        self.gripper_actuator_id = (
            self.model.actuator(
                GRIPPER_ACTUATOR
            ).id
        )

        self.gripper_qpos_id = (
            self.model.jnt_qposadr[
                self.model.joint(
                    GRIPPER_JOINT
                ).id
            ]
        )

        self.tcp_id = self.model.site(
            "gripper_pinch"
        ).id

        self.cube_id = self.model.body(
            "pick_cube"
        ).id

        self.target_id = self.model.site(
            "place_target"
        ).id

        # Cubeのfreejointを取得する
        cube_joint_id = int(
            self.model.body_jntadr[
                self.cube_id
            ]
        )

        if cube_joint_id < 0:
            raise RuntimeError(
                "pick_cube has no joint"
            )

        if (
            self.model.jnt_type[cube_joint_id]
            != mujoco.mjtJoint.mjJNT_FREE
        ):
            raise RuntimeError(
                "pick_cube must have a freejoint"
            )

        self.cube_qpos_id = int(
            self.model.jnt_qposadr[
                cube_joint_id
            ]
        )

        table = self.model.geom(
            "worktable"
        )

        self.table_top = float(
            table.pos[2] + table.size[2]
        )

        self.arm_ctrl_low = (
            self.model.actuator_ctrlrange[
                self.arm_actuator_ids, 0
            ]
        )

        self.arm_ctrl_high = (
            self.model.actuator_ctrlrange[
                self.arm_actuator_ids, 1
            ]
        )

        self.steps_per_control = round(
            CONTROL_DT
            / self.model.opt.timestep
        )

        if not np.isclose(
            self.steps_per_control
            * self.model.opt.timestep,
            CONTROL_DT,
        ):
            raise ValueError(
                "Invalid control timestep"
            )

        self.num_control_steps = round(
            PickPlaceControllerV005.T_END
            / CONTROL_DT
        )

        self.renderer = mujoco.Renderer(
            self.model,
            height=IMAGE_HEIGHT,
            width=IMAGE_WIDTH,
        )

        # ランダム化前の元のキューブ位置
        self.reset_simulation()

        self.base_cube_position = (
            self.data.xpos[self.cube_id].copy()
        )

    def reset_simulation(
        self,
        cube_offset_xy=None,
    ):
        """1エピソードの初期状態に戻す。"""

        mujoco.mj_resetData(
            self.model,
            self.data,
        )

        self.data.qpos[
            self.arm_qpos_ids
        ] = HOME_Q

        self.data.ctrl[
            self.arm_actuator_ids
        ] = HOME_Q

        self.data.ctrl[
            self.gripper_actuator_id
        ] = 0.0

        if cube_offset_xy is not None:
            offset = np.asarray(
                cube_offset_xy,
                dtype=float,
            )

            # freejointのqpos先頭3成分がXYZ位置
            self.data.qpos[
                self.cube_qpos_id:
                self.cube_qpos_id + 2
            ] += offset

        mujoco.mj_forward(
            self.model,
            self.data,
        )

    def collect_episode(
        self,
        episode_id: str,
        cube_offset_xy: np.ndarray,
    ) -> dict:
        """1件の教師エピソードを生成・保存する。"""

        self.reset_simulation(cube_offset_xy)

        home_position = (
            self.data.site_xpos[
                self.tcp_id
            ].copy()
        )

        home_rotation = (
            self.data.site_xmat[
                self.tcp_id
            ].copy().reshape(3, 3)
        )

        cube_initial = (
            self.data.xpos[
                self.cube_id
            ].copy()
        )

        place_target = (
            self.data.site_xpos[
                self.target_id
            ].copy()
        )

        controller = PickPlaceControllerV005(
            home_position=home_position,
            cube_position=cube_initial,
            place_target=place_target,
            table_top=self.table_top,
        )

        collector = EpisodeCollector(
            output_dir=(
                OUTPUT_ROOT / episode_id
            ),
            camera_name=CAMERA_NAME,
        )

        place_configured = False
        max_cube_clearance = -np.inf

        start_time = time.perf_counter()

        for step in range(self.num_control_steps):
            # 整数ステップから教師制御時刻を計算
            t = step * CONTROL_DT


            spec = getattr(self, "perturb_spec", None)

            if spec is not None and step == spec["step"]:
                actual = []
                for j, delta in enumerate(spec["delta"]):
                    qid = self.arm_qpos_ids[j]
                    vid = self.arm_qvel_ids[j]
                    jid = self.model.joint(ARM_JOINTS[j]).id

                    old_q = float(self.data.qpos[qid])
                    new_q = old_q + float(delta)

                    if self.model.jnt_limited[jid]:
                        low, high = self.model.jnt_range[jid]
                        new_q = float(np.clip(new_q, low, high))

                    self.data.qpos[qid] = new_q
                    self.data.qvel[vid] = 0.0
                    actual.append(new_q - old_q)

                mujoco.mj_forward(self.model, self.data)
                print(
                    "[Perturbation]",
                    f"time={t:.2f}s",
                    f"delta={np.round(actual, 5).tolist()}",
                )

            # 実際のMuJoCo観測時刻
            observation_time = float(
                self.data.time
            )

            # Transfer終了時の実測位置でPlace位置を調整
            if (
                not place_configured
                and t >= controller.T_TRANSFER_HOLD
            ):
                controller.configure_place(
                    tcp_position=(
                        self.data.site_xpos[
                            self.tcp_id
                        ].copy()
                    ),
                    cube_position=(
                        self.data.xpos[
                            self.cube_id
                        ].copy()
                    ),
                )

                place_configured = True

            task_action = controller.get_action(t)

            # TCP目標から関節角度を算出
            target_q = self.ik.solve(
                target_position=(
                    task_action.target_position
                ),
                target_rotation=home_rotation,
                current_qpos=self.data.qpos.copy(),
            )

            self.data.ctrl[
                self.arm_actuator_ids
            ] = np.clip(
                target_q,
                self.arm_ctrl_low,
                self.arm_ctrl_high,
            )

            self.data.ctrl[
                self.gripper_actuator_id
            ] = task_action.gripper_command

            teacher_action = np.concatenate([
                self.data.ctrl[
                    self.arm_actuator_ids
                ].copy(),
                [
                    float(
                        self.data.ctrl[
                            self.gripper_actuator_id
                        ]
                    )
                ],
            ])

            # 現在の観測と同じ状態から画像取得
            self.renderer.update_scene(
                self.data,
                camera=CAMERA_NAME,
            )

            rgb = self.renderer.render().copy()

            collector.append(
                timestamp=observation_time,
                joint_position=(
                    self.data.qpos[
                        self.arm_qpos_ids
                    ].copy()
                ),
                joint_velocity=(
                    self.data.qvel[
                        self.arm_qvel_ids
                    ].copy()
                ),
                gripper_position=float(
                    self.data.qpos[
                        self.gripper_qpos_id
                    ]
                ),
                action=teacher_action,
                tcp_position=(
                    self.data.site_xpos[
                        self.tcp_id
                    ].copy()
                ),
                cube_position=(
                    self.data.xpos[
                        self.cube_id
                    ].copy()
                ),
                phase=task_action.phase,
                rgb=rgb,
            )

            # 次の観測へ物理時間を進める
            for _ in range(self.steps_per_control):
                mujoco.mj_step(
                    self.model,
                    self.data,
                )

            # 持ち上げ・運搬中の高さを記録
            if (
                controller.T_LIFT
                <= t
                < controller.T_PLACE
            ):
                cube_z = float(
                    self.data.xpos[
                        self.cube_id, 2
                    ]
                )

                clearance = (
                    cube_z
                    - controller.CUBE_HALF_SIZE
                    - self.table_top
                )

                max_cube_clearance = max(
                    max_cube_clearance,
                    clearance,
                )

            if (step + 1) % 500 == 0:
                print(
                    f"  {step + 1}"
                    f"/{self.num_control_steps}"
                    " steps"
                )

        # ----------------------------------------------------
        # 最終評価
        # ----------------------------------------------------
        cube_final = (
            self.data.xpos[
                self.cube_id
            ].copy()
        )

        xy_error = float(
            np.linalg.norm(
                cube_final[:2]
                - place_target[:2]
            )
        )

        cube_bottom = (
            cube_final[2]
            - controller.CUBE_HALF_SIZE
        )

        height_error = float(
            abs(
                cube_bottom
                - self.table_top
            )
        )

        lift_success = (
            max_cube_clearance > 0.05
        )

        xy_success = xy_error < 0.03
        height_success = height_error < 0.015

        success = bool(
            lift_success
            and xy_success
            and height_success
        )

        # ----------------------------------------------------
        # 保存
        # ----------------------------------------------------
        metadata = {
            "episode_id": episode_id,
            "simulator": "MuJoCo",
            "robot": "UR5e",
            "gripper": "Robotiq 2F-85",
            "task": "pick_and_place",
            "model_path": str(
                MODEL_PATH.relative_to(ROOT)
            ),
            "control_dt": CONTROL_DT,
            "joint_names": list(ARM_JOINTS),
            "action_joint_names": list(
                ARM_ACTUATORS
            ),
            "gripper_actuator": (
                GRIPPER_ACTUATOR
            ),
            "gripper_observation_joint": (
                GRIPPER_JOINT
            ),
            "gripper_observation_unit": "rad",
            "action_dimension": 7,
            "camera_width": IMAGE_WIDTH,
            "camera_height": IMAGE_HEIGHT,
            "cube_initial_position": (
                cube_initial.tolist()
            ),
            "cube_offset_xy_m": (
                cube_offset_xy.tolist()
            ),
            "place_target_position": (
                place_target.tolist()
            ),
            "xy_error_m": xy_error,
            "height_error_m": height_error,
            "max_cube_clearance_m": (
                float(max_cube_clearance)
            ),
            "lift_success": bool(lift_success),
            "xy_success": bool(xy_success),
            "height_success": bool(
                height_success
            ),
            "completed": True,
        }

        saved_path = collector.save(
            metadata=metadata,
            success=success,
        )

        elapsed = (
            time.perf_counter()
            - start_time
        )

        return {
            "episode_id": episode_id,
            "success": success,
            "xy_error_mm": xy_error * 1000.0,
            "height_error_mm": (
                height_error * 1000.0
            ),
            "max_clearance_mm": (
                max_cube_clearance * 1000.0
            ),
            "elapsed_sec": elapsed,
            "saved_path": str(saved_path),
        }

    def close(self):
        self.renderer.close()


def main():
    args = parse_args()

    global OUTPUT_ROOT
    OUTPUT_ROOT = args.output_dir.resolve()

    if args.episodes <= 0:
        raise ValueError(
            "--episodes must be positive"
        )

    if args.range_mm < 0:
        raise ValueError(
            "--range-mm must be >= 0"
        )

    if args.start_id <= 0:
        raise ValueError(
            "--start-id must be positive"
        )

    episode_ids = [
        f"episode_{number:04d}"
        for number in range(
            args.start_id,
            args.start_id + args.episodes,
        )
    ]

    # 既存データを誤って上書きしない
    existing = [
        name
        for name in episode_ids
        if (OUTPUT_ROOT / name).exists()
    ]

    if existing:
        raise FileExistsError(
            "Episode directories already exist: "
            + ", ".join(existing)
            + ". Change --start-id."
        )

    rng = np.random.default_rng(
        args.seed
    )

    random_range_m = (
        args.range_mm / 1000.0
    )

    print("\n=== Demonstration Collection ===")
    print("Episodes:", args.episodes)
    print("Start ID:", args.start_id)
    print("Seed:", args.seed)
    print("XY range: +/-", args.range_mm, "mm")
    print("Control dt:", CONTROL_DT)
    print("Camera:", CAMERA_NAME)
    print("Model:", MODEL_PATH)

    runner = DemonstrationRunner()
    results = []

    try:
        for episode_id in episode_ids:
            offset = rng.uniform(
                low=-random_range_m,
                high=random_range_m,
                size=2,
            )

            print(
                f"\n--- {episode_id} ---"
            )

            print(
                "Cube offset XY (mm):",
                np.round(
                    offset * 1000.0,
                    2,
                ),
            )


            perturb_spec = None

            if args.perturb_mode != "none":
                if not (0 < args.perturb_rad <= 0.1):
                    raise ValueError("--perturb-rad must be in (0, 0.1]")

                pt = (
                    args.perturb_time
                    if args.perturb_time is not None
                    else float(rng.uniform(3.0, 6.0))
                )
                if not (0 <= pt < runner.num_control_steps * CONTROL_DT):
                    raise ValueError("Invalid perturbation time")

                delta = np.zeros(6, dtype=float)

                if args.perturb_mode == "single":
                    j = int(rng.integers(0, 6))
                    delta[j] = float(
                        rng.choice([-1.0, 1.0]) * args.perturb_rad
                    )
                else:
                    delta = (
                        rng.choice([-1.0, 1.0], size=6)
                        * rng.uniform(0.5, 1.0, size=6)
                        * args.perturb_rad
                    )

                perturb_spec = {
                    "mode": args.perturb_mode,
                    "step": round(pt / CONTROL_DT),
                    "delta": delta.tolist(),
                }

            runner.perturb_spec = perturb_spec

            result = runner.collect_episode(
                episode_id=episode_id,
                cube_offset_xy=offset,
            )


            import json

            settings = {
                "generator": "experiment_006b",
                "seed": args.seed,
                "cube_offset_xy_mm": (offset * 1000).tolist(),
                "perturbation": perturb_spec,
                "success": bool(result["success"]),
            }
            settings_path = OUTPUT_ROOT / episode_id / "v006b_settings.json"
            settings_path.write_text(
                json.dumps(settings, indent=2),
                encoding="utf-8",
            )

            results.append(result)

            print(
                "Success:",
                result["success"],
            )
            print(
                "XY error:",
                round(
                    result["xy_error_mm"],
                    2,
                ),
                "mm",
            )
            print(
                "Height error:",
                round(
                    result["height_error_mm"],
                    2,
                ),
                "mm",
            )
            print(
                "Max clearance:",
                round(
                    result["max_clearance_mm"],
                    2,
                ),
                "mm",
            )

    finally:
        runner.close()

    # --------------------------------------------------------
    # 全エピソードの集計
    # --------------------------------------------------------
    success_count = sum(
        result["success"]
        for result in results
    )

    total_count = len(results)

    success_rate = (
        100.0
        * success_count
        / total_count
    )

    print("\n=== Collection Summary ===")
    print("Total episodes:", total_count)
    print("Successful:", success_count)
    print(
        "Failed:",
        total_count - success_count,
    )
    print(
        "Success rate:",
        f"{success_rate:.1f}%",
    )

    for result in results:
        print(
            result["episode_id"],
            "SUCCESS" if result["success"] else "FAILED",
            f"XY={result['xy_error_mm']:.2f}mm",
        )


if __name__ == "__main__":
    main()
