
from dataclasses import dataclass

import numpy as np


@dataclass
class TaskAction:
    phase: str
    target_position: np.ndarray
    gripper_command: float


class PickPlaceController:
    """Pick & Placeの状態遷移とTCP目標軌道を生成する。"""

    CONTROL_DT = 0.02
    LIFT_HEIGHT = 0.10
    CUBE_HALF_SIZE = 0.025
    PLACE_CLEARANCE = 0.002

    # 各フェーズの終了時刻（秒）
    T_APPROACH = 3.0
    T_DESCEND = 6.0
    T_GRASP = 8.0
    T_LIFT = 11.0
    T_HOLD = 13.0
    T_TRANSFER = 17.0
    T_TRANSFER_HOLD = 19.0
    T_PLACE = 22.0
    T_RELEASE = 24.0
    T_SETTLE = 25.0
    T_RETREAT = 28.0
    T_END = 29.0

    def __init__(
        self,
        home_position: np.ndarray,
        cube_position: np.ndarray,
        place_target: np.ndarray,
        table_top: float,
    ):
        self.home = np.asarray(
            home_position, dtype=float
        ).copy()

        self.cube = np.asarray(
            cube_position, dtype=float
        ).copy()

        self.place_target = np.asarray(
            place_target, dtype=float
        ).copy()

        self.table_top = float(table_top)

        self.pre_grasp = self.cube + np.array([
            0.0, 0.0, 0.12
        ])

        self.grasp = self.cube.copy()

        self.lift = self.grasp + np.array([
            0.0, 0.0, self.LIFT_HEIGHT
        ])

        self.transfer = np.array([
            self.place_target[0],
            self.place_target[1],
            self.lift[2],
        ])

        self.place_tcp = None
        self.retreat_tcp = None

    @staticmethod
    def smoothstep(u: float) -> float:
        u = float(np.clip(u, 0.0, 1.0))
        return 3.0 * u**2 - 2.0 * u**3

    @classmethod
    def interpolate(
        cls,
        start: np.ndarray,
        end: np.ndarray,
        u: float,
    ) -> np.ndarray:
        s = cls.smoothstep(u)
        return (1.0 - s) * start + s * end

    def configure_place(
        self,
        tcp_position: np.ndarray,
        cube_position: np.ndarray,
    ) -> None:
        """運搬完了時のTCPとキューブのずれから配置位置を決定する。"""

        tcp_position = np.asarray(
            tcp_position, dtype=float
        )
        cube_position = np.asarray(
            cube_position, dtype=float
        )

        tcp_offset = tcp_position - cube_position

        desired_cube = np.array([
            self.place_target[0],
            self.place_target[1],
            self.table_top
            + self.CUBE_HALF_SIZE
            + self.PLACE_CLEARANCE,
        ])

        self.place_tcp = desired_cube + tcp_offset

        self.retreat_tcp = (
            self.place_tcp
            + np.array([0.0, 0.0, 0.12])
        )

    def get_action(self, t: float) -> TaskAction:
        """時刻から教師のTCP目標位置とグリッパー指令を返す。"""

        if t < self.T_APPROACH:
            return TaskAction(
                "APPROACH",
                self.interpolate(
                    self.home,
                    self.pre_grasp,
                    t / self.T_APPROACH,
                ),
                0.0,
            )

        if t < self.T_DESCEND:
            return TaskAction(
                "DESCEND",
                self.interpolate(
                    self.pre_grasp,
                    self.grasp,
                    (t - self.T_APPROACH)
                    / (self.T_DESCEND - self.T_APPROACH),
                ),
                0.0,
            )

        if t < self.T_GRASP:
            u = (
                (t - self.T_DESCEND)
                / (self.T_GRASP - self.T_DESCEND)
            )

            return TaskAction(
                "GRASP",
                self.grasp.copy(),
                255.0 * self.smoothstep(u),
            )

        if t < self.T_LIFT:
            return TaskAction(
                "LIFT",
                self.interpolate(
                    self.grasp,
                    self.lift,
                    (t - self.T_GRASP)
                    / (self.T_LIFT - self.T_GRASP),
                ),
                255.0,
            )

        if t < self.T_HOLD:
            return TaskAction(
                "HOLD",
                self.lift.copy(),
                255.0,
            )

        if t < self.T_TRANSFER:
            return TaskAction(
                "TRANSFER",
                self.interpolate(
                    self.lift,
                    self.transfer,
                    (t - self.T_HOLD)
                    / (self.T_TRANSFER - self.T_HOLD),
                ),
                255.0,
            )

        if t < self.T_TRANSFER_HOLD:
            return TaskAction(
                "TRANSFER_HOLD",
                self.transfer.copy(),
                255.0,
            )

        if self.place_tcp is None:
            raise RuntimeError(
                "Call configure_place() before PLACE"
            )

        if t < self.T_PLACE:
            return TaskAction(
                "PLACE",
                self.interpolate(
                    self.transfer,
                    self.place_tcp,
                    (t - self.T_TRANSFER_HOLD)
                    / (self.T_PLACE - self.T_TRANSFER_HOLD),
                ),
                255.0,
            )

        if t < self.T_RELEASE:
            u = (
                (t - self.T_PLACE)
                / (self.T_RELEASE - self.T_PLACE)
            )

            return TaskAction(
                "RELEASE",
                self.place_tcp.copy(),
                255.0 * (1.0 - self.smoothstep(u)),
            )

        if t < self.T_SETTLE:
            return TaskAction(
                "SETTLE",
                self.place_tcp.copy(),
                0.0,
            )

        if t < self.T_RETREAT:
            return TaskAction(
                "RETREAT",
                self.interpolate(
                    self.place_tcp,
                    self.retreat_tcp,
                    (t - self.T_SETTLE)
                    / (self.T_RETREAT - self.T_SETTLE),
                ),
                0.0,
            )

        return TaskAction(
            "DONE",
            self.retreat_tcp.copy(),
            0.0,
        )
