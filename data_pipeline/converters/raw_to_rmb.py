
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]

RAW_ROOT = ROOT / "data" / "raw"
RMB_ROOT = ROOT / "data" / "rmb" / "pick_and_place"

CAMERA_NAME = "front"
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240

REQUIRED_KEYS = (
    "timestamp",
    "joint_position",
    "gripper_position",
    "action",
)

RMB_KEYS = (
    "time",
    "measured_joint_pos",
    "command_joint_pos",
    "front_rgb_image",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert MuJoCo RAW episodes to RMB HDF5"
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=RAW_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=RMB_ROOT,
    )
    parser.add_argument(
        "--include-failed",
        action="store_true",
        help="Include unsuccessful episodes",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing HDF5 files",
    )
    return parser.parse_args()


def load_raw_episode(episode_dir: Path):
    npz_path = episode_dir / "episode.npz"
    metadata_path = episode_dir / "metadata.json"

    if not npz_path.is_file():
        raise FileNotFoundError(npz_path)

    if not metadata_path.is_file():
        raise FileNotFoundError(metadata_path)

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    with np.load(
        npz_path,
        allow_pickle=False,
    ) as raw:
        missing = set(REQUIRED_KEYS) - set(raw.files)

        if missing:
            raise ValueError(
                f"Missing RAW keys: {sorted(missing)}"
            )

        timestamp = raw["timestamp"].copy()
        joint_position = raw["joint_position"].copy()
        gripper_position = raw["gripper_position"].copy()
        action = raw["action"].copy()

    steps = len(timestamp)

    if steps == 0:
        raise ValueError("Empty episode")

    if timestamp.shape != (steps,):
        raise ValueError("Invalid timestamp shape")

    if joint_position.shape != (steps, 6):
        raise ValueError("Invalid joint_position shape")

    if gripper_position.shape != (steps,):
        raise ValueError("Invalid gripper_position shape")

    if action.shape != (steps, 7):
        raise ValueError("Invalid action shape")

    if not np.all(np.diff(timestamp) > 0):
        raise ValueError("Non-monotonic timestamps")

    for name, array in {
        "timestamp": timestamp,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
        "action": action,
    }.items():
        if not np.all(np.isfinite(array)):
            raise ValueError(f"Non-finite data: {name}")

    if metadata.get("num_steps") != steps:
        raise ValueError(
            "metadata num_steps does not match RAW"
        )

    if metadata.get("camera_name") != CAMERA_NAME:
        raise ValueError(
            "Unexpected camera name in metadata"
        )

    if metadata.get("image_count") != steps:
        raise ValueError(
            "metadata image_count does not match RAW"
        )

    image_shape = metadata.get("image_shape")

    if image_shape != [
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        3,
    ]:
        raise ValueError(
            f"Unexpected image_shape: {image_shape}"
        )

    image_dir = (
        episode_dir / "images" / CAMERA_NAME
    )
    image_paths = sorted(
        image_dir.glob("*.png")
    )

    if len(image_paths) != steps:
        raise ValueError(
            f"Image count mismatch: "
            f"{len(image_paths)} != {steps}"
        )

    # フレーム番号が連続していることを確認
    for index, image_path in enumerate(image_paths):
        expected_name = f"{index:06d}.png"
        if image_path.name != expected_name:
            raise ValueError(
                f"Unexpected image filename: "
                f"{image_path.name} "
                f"(expected {expected_name})"
            )

    measured_joint_pos = np.concatenate(
        [
            joint_position,
            gripper_position[:, None],
        ],
        axis=1,
    ).astype(np.float64)

    command_joint_pos = action.astype(np.float64)

    return {
        "time": timestamp.astype(np.float64),
        "measured_joint_pos": measured_joint_pos,
        "command_joint_pos": command_joint_pos,
        "image_paths": image_paths,
        "metadata": metadata,
    }


def validate_hdf5(path: Path, expected_steps: int):
    """書き出し後のHDF5を読み直して検証する。"""

    with h5py.File(path, "r") as h5:
        missing = set(RMB_KEYS) - set(h5.keys())

        if missing:
            raise ValueError(
                f"Missing RMB keys: {sorted(missing)}"
            )

        expected_shapes = {
            "time": (expected_steps,),
            "measured_joint_pos": (expected_steps, 7),
            "command_joint_pos": (expected_steps, 7),
            "front_rgb_image": (
                expected_steps,
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3,
            ),
        }

        for key, shape in expected_shapes.items():
            if h5[key].shape != shape:
                raise ValueError(
                    f"{key}: {h5[key].shape} != {shape}"
                )

        if h5["front_rgb_image"].dtype != np.uint8:
            raise ValueError("Invalid RGB dtype")

        for key in (
            "time",
            "measured_joint_pos",
            "command_joint_pos",
        ):
            if not np.all(np.isfinite(h5[key][:])):
                raise ValueError(
                    f"Non-finite RMB data: {key}"
                )

        if not np.all(np.diff(h5["time"][:]) > 0):
            raise ValueError("Invalid RMB timestamps")

        # 先頭・末尾フレームを読み出せるか確認
        for index in (0, expected_steps - 1):
            frame = h5["front_rgb_image"][index]
            if frame.shape != (
                IMAGE_HEIGHT,
                IMAGE_WIDTH,
                3,
            ):
                raise ValueError("Invalid RGB frame")


def convert_episode(
    episode_dir: Path,
    output_dir: Path,
    overwrite: bool,
):
    episode_id = episode_dir.name
    output_path = output_dir / f"{episode_id}.hdf5"

    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}"
        )

    episode = load_raw_episode(episode_dir)

    steps = len(episode["time"])

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 途中失敗で完成ファイルを壊さないようにする
    temp_path = output_dir / f"{episode_id}.hdf5.tmp"

    if temp_path.exists():
        raise FileExistsError(
            f"Temporary file exists: {temp_path}"
        )

    try:
        with h5py.File(temp_path, "w") as h5:
            # 数値データ
            for key in (
                "time",
                "measured_joint_pos",
                "command_joint_pos",
            ):
                h5.create_dataset(
                    key,
                    data=episode[key],
                )

            # RGB画像は1フレームずつ書き込む
            # 全フレームをメモリに載せない
            rgb_dataset = h5.create_dataset(
                "front_rgb_image",
                shape=(
                    steps,
                    IMAGE_HEIGHT,
                    IMAGE_WIDTH,
                    3,
                ),
                dtype=np.uint8,
                chunks=(1, IMAGE_HEIGHT, IMAGE_WIDTH, 3),
                compression="gzip",
                compression_opts=1,
            )

            for index, image_path in enumerate(
                episode["image_paths"]
            ):
                with Image.open(image_path) as img:
                    if img.mode != "RGB":
                        raise ValueError(
                            f"Not RGB: {image_path}"
                        )

                    if img.size != (
                        IMAGE_WIDTH,
                        IMAGE_HEIGHT,
                    ):
                        raise ValueError(
                            f"Invalid image size: {image_path}"
                        )

                    rgb = np.asarray(
                        img,
                        dtype=np.uint8,
                    )

                    rgb_dataset[index] = rgb

                if (index + 1) % 500 == 0:
                    print(
                        f"  RGB frames: "
                        f"{index + 1}/{steps}"
                    )

            # RMB互換メタデータ
            h5.attrs["format"] = (
                "RmbData-SingleHDF5"
            )
            h5.attrs["demo_name"] = episode_id
            h5.attrs["task_desc"] = "pick_and_place"
            h5.attrs["camera_names"] = CAMERA_NAME
            h5.attrs["source"] = "manipulation_lab"
            h5.attrs["success"] = bool(
                episode["metadata"]["success"]
            )

        validate_hdf5(
            temp_path,
            expected_steps=steps,
        )

        os.replace(temp_path, output_path)

    finally:
        # 変換失敗時だけ一時ファイルを削除
        if temp_path.exists():
            temp_path.unlink()

    return output_path, steps


def main():
    args = parse_args()

    episode_dirs = sorted(
        path
        for path in args.input_dir.glob("episode_*")
        if path.is_dir()
    )

    if not episode_dirs:
        raise FileNotFoundError(
            f"No RAW episodes: {args.input_dir}"
        )

    converted = 0
    skipped = 0
    total_steps = 0

    print("\n=== RAW to RMB Conversion ===")
    print("Input:", args.input_dir)
    print("Output:", args.output_dir)

    for episode_dir in episode_dirs:
        metadata_path = (
            episode_dir / "metadata.json"
        )

        metadata = json.loads(
            metadata_path.read_text(encoding="utf-8")
        )

        if (
            not args.include_failed
            and not metadata.get("success", False)
        ):
            print(
                f"SKIP {episode_dir.name}: "
                "unsuccessful episode"
            )
            skipped += 1
            continue

        output_path = (
            args.output_dir / f"{episode_dir.name}.hdf5"
        )

        if output_path.exists() and not args.overwrite:
            print(
                f"SKIP {episode_dir.name}: "
                "already converted"
            )
            skipped += 1
            continue

        print(f"\n--- {episode_dir.name} ---")

        output_path, steps = convert_episode(
            episode_dir=episode_dir,
            output_dir=args.output_dir,
            overwrite=args.overwrite,
        )

        print("Saved:", output_path)
        print("Steps:", steps)
        print("Validation: OK")

        converted += 1
        total_steps += steps

    print("\n=== Conversion Summary ===")
    print("Converted episodes:", converted)
    print("Skipped episodes:", skipped)
    print("Total steps:", total_steps)


if __name__ == "__main__":
    main()
