
import json
from pathlib import Path

import numpy as np
from PIL import Image


class EpisodeCollector:
    """MuJoCoのRAW観測・教師アクション・画像を保存する。"""

    def __init__(
        self,
        output_dir: Path,
        camera_name: str = "front",
    ):
        self.output_dir = Path(output_dir)
        self.camera_name = camera_name

        self.image_dir = (
            self.output_dir / "images" / camera_name
        )
        self.image_dir.mkdir(parents=True, exist_ok=True)

        # 同じepisode_idで再実行した場合の古い画像を除去
        for old_file in self.image_dir.glob("*.png"):
            old_file.unlink()

        self.records = {
            "timestamp": [],
            "joint_position": [],
            "joint_velocity": [],
            "gripper_position": [],
            "action": [],
            "tcp_position": [],
            "cube_position": [],
            "phase": [],
        }

        self.image_count = 0
        self.image_shape = None

    def append(
        self,
        timestamp: float,
        joint_position: np.ndarray,
        joint_velocity: np.ndarray,
        gripper_position: float,
        action: np.ndarray,
        tcp_position: np.ndarray,
        cube_position: np.ndarray,
        phase: str,
        rgb: np.ndarray,
    ) -> None:
        """同一時刻の観測・アクション・RGB画像を記録する。"""

        rgb = np.asarray(rgb)

        if rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError(
                f"RGB must have shape (H,W,3): {rgb.shape}"
            )

        if rgb.dtype != np.uint8:
            raise ValueError(
                f"RGB dtype must be uint8: {rgb.dtype}"
            )

        if self.image_shape is None:
            self.image_shape = tuple(rgb.shape)
        elif tuple(rgb.shape) != self.image_shape:
            raise ValueError("Inconsistent image shape")

        values = {
            "timestamp": float(timestamp),
            "joint_position": joint_position,
            "joint_velocity": joint_velocity,
            "gripper_position": float(gripper_position),
            "action": action,
            "tcp_position": tcp_position,
            "cube_position": cube_position,
            "phase": str(phase),
        }

        # 数値レコードを検証してから保存
        for key, value in values.items():
            if key == "phase":
                continue
            if not np.all(np.isfinite(value)):
                raise ValueError(f"Non-finite value: {key}")

        frame_index = len(self.records["timestamp"])

        image_path = (
            self.image_dir / f"{frame_index:06d}.png"
        )
        Image.fromarray(rgb).save(image_path)

        for key, value in values.items():
            if key == "phase":
                self.records[key].append(value)
            else:
                self.records[key].append(
                    np.array(value, copy=True)
                )

        self.image_count += 1

    def save(
        self,
        metadata: dict,
        success: bool,
    ) -> Path:
        """整合性を検証してNPZ・JSONを保存する。"""

        arrays = {
            key: np.asarray(values)
            for key, values in self.records.items()
        }

        lengths = {len(x) for x in arrays.values()}

        if len(lengths) != 1 or not lengths or 0 in lengths:
            raise ValueError("Invalid episode lengths")

        num_steps = len(arrays["timestamp"])

        if self.image_count != num_steps:
            raise ValueError(
                "RGB frame count does not match observations"
            )

        timestamps = arrays["timestamp"]

        if not np.all(np.diff(timestamps) > 0):
            raise ValueError("Non-monotonic timestamps")

        expected_shapes = {
            "joint_position": (6,),
            "joint_velocity": (6,),
            "gripper_position": (),
            "action": (7,),
            "tcp_position": (3,),
            "cube_position": (3,),
        }

        for key, shape in expected_shapes.items():
            if arrays[key].shape[1:] != shape:
                raise ValueError(
                    f"{key}: unexpected shape "
                    f"{arrays[key].shape}"
                )

        for key, array in arrays.items():
            if key == "phase":
                continue

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"Non-finite values in {key}"
                )

        output_file = self.output_dir / "episode.npz"
        np.savez_compressed(output_file, **arrays)

        info = {
            **metadata,
            "success": bool(success),
            "num_steps": num_steps,
            "record_keys": list(arrays.keys()),
            "action_type": "target_joint_position",
            "gripper_action_unit": "raw_0_255",
            "timestamp_unit": "seconds",
            "sample_alignment": "pre_step_observation_action",
            "camera_name": self.camera_name,
            "image_shape": list(self.image_shape),
            "image_count": self.image_count,
            "image_directory": (
                f"images/{self.camera_name}"
            ),
            "image_format": "png",
            "image_index_start": 0,
        }

        metadata_path = (
            self.output_dir / "metadata.json"
        )

        metadata_path.write_text(
            json.dumps(
                info,
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        return output_file
