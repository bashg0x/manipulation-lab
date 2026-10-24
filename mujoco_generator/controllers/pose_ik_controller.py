import mujoco
import numpy as np


ARM_JOINTS = (
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
)


class PoseIKController:

    def __init__(
        self,
        model: mujoco.MjModel,
        site_name: str = "gripper_pinch",
        damping: float = 0.05,
    ):
        self.model = model
        self.data = mujoco.MjData(model)
        self.site_id = model.site(site_name).id
        self.damping = damping

        joint_ids = [
            model.joint(name).id
            for name in ARM_JOINTS
        ]

        self.qpos_ids = model.jnt_qposadr[joint_ids]
        self.dof_ids = model.jnt_dofadr[joint_ids]
        self.joint_ids = joint_ids

    def solve(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray,
        current_qpos: np.ndarray,
        max_iterations: int = 150,
        position_tolerance: float = 0.001,
        rotation_tolerance: float = 0.01,
    ):
        self.data.qpos[:] = current_qpos
        self.data.qvel[:] = 0

        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))

        target_quat = np.zeros(4)
        mujoco.mju_mat2Quat(
            target_quat,
            np.asarray(target_rotation).reshape(9),
        )

        for _ in range(max_iterations):
            mujoco.mj_forward(self.model, self.data)

            current_position = (
                self.data.site_xpos[self.site_id].copy()
            )

            current_rotation = (
                self.data.site_xmat[self.site_id].copy()
            )

            position_error = (
                target_position - current_position
            )

            current_quat = np.zeros(4)
            mujoco.mju_mat2Quat(
                current_quat,
                current_rotation.reshape(9),
            )

            rotation_error_local = np.zeros(3)

            mujoco.mju_subQuat(
                rotation_error_local,
                target_quat,
                current_quat,
            )

            # 現在のTCP姿勢を使い、
            # 回転誤差をワールド座標系へ変換する
            rotation_error = (
                current_rotation.reshape(3, 3)
                @ rotation_error_local
            )

            if (
                np.linalg.norm(position_error)
                < position_tolerance
                and np.linalg.norm(rotation_error)
                < rotation_tolerance
            ):
                break

            mujoco.mj_jacSite(
                self.model,
                self.data,
                jacp,
                jacr,
                self.site_id,
            )

            J = np.vstack([
                jacp[:, self.dof_ids],
                jacr[:, self.dof_ids],
            ])

            error = np.concatenate([
                position_error,
                rotation_error,
            ])

            dq = J.T @ np.linalg.solve(
                J @ J.T + self.damping**2 * np.eye(6),
                error,
            )

            dq = np.clip(dq, -0.03, 0.03)

            self.data.qpos[self.qpos_ids] += dq

            for joint_id, qpos_id in zip(
                self.joint_ids, self.qpos_ids
            ):
                if self.model.jnt_limited[joint_id]:
                    low, high = self.model.jnt_range[joint_id]
                    self.data.qpos[qpos_id] = np.clip(
                        self.data.qpos[qpos_id],
                        low,
                        high,
                    )

        mujoco.mj_forward(self.model, self.data)

        return self.data.qpos[self.qpos_ids].copy()
