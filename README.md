# Manipulation Lab — MuJoCo Pick & Place with ACT

A reproducible Physical AI coursework experiment combining **MuJoCo**, demonstration collection, **RoboManipBaselines (RMB)**, and **Action Chunking with Transformers (ACT)**. The project studies the difference between **fully autonomous ACT control** and **teacher-assisted execution followed by ACT control**.

> **Key result:** ACT-only execution did **not** complete Pick & Place in any of the three recorded test conditions. Switching from the V005 teacher to ACT at 15 seconds met the project's *provisional geometric success criterion* in all three conditions. These are different evaluation modes; teacher-assisted success is **not** ACT-only success.

## Workflow

```text
MuJoCo UR5e + Robotiq gripper
       │
       ▼
Demonstration generation (teacher / perturbations / DAgger)
       │  RAW episode.npz + camera frames + metadata
       ▼
RAW → RMB HDF5 conversion
       │
       ▼
Fixed train/validation split (80 / 20 episodes)
       │
       ▼
ACT training on Google Colab GPU
       │
       ▼
MuJoCo evaluation: ACT Only vs Teacher → ACT
       │
       ▼
Four-panel comparison video
```

## Environment

- Host: Windows 11 with WSL2 / Ubuntu 24.04
- Python: 3.12
- Physics simulator: MuJoCo
- Learning framework: RoboManipBaselines with its ACT dependency, pinned by the Git submodule
- Other Python packages used by the local pipeline: `numpy`, `mujoco`, `h5py`, `Pillow`, `torch`, `torchvision`
- Video compositor: FFmpeg / ffprobe
- GPU training: Google Colab (CUDA); evaluation can run on CPU

Clone with submodules:

```bash
git clone --recurse-submodules <YOUR_REPOSITORY_URL>
cd manipulation_lab
```

Create a Python 3.12 virtual environment and install the compatible dependencies for the pinned RMB revision. See [environment setup](docs/01_manipulation_lab_setup.md) and the Colab notebook for environment-specific setup. There is currently no verified standalone lockfile for the local environment; package names above are not an exact version lock.

From the repository root, use:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Repository layout

```text
app/RoboManipBaselines/                        # Pinned Git submodule
mujoco_generator/
  assets/                                     # UR5e / Robotiq meshes and third-party notices
  environments/pick_and_place_camera.xml     # MuJoCo scene
  controllers/                               # Teacher and IK controllers
  collectors/episode_collector.py
  scripts/collect_demonstrations_v006b.py

data_pipeline/converters/raw_to_rmb.py
training/
  scripts/train_act_v006b.py
  experiment_006b_split.json
notebooks/train_act_colab_v006b.ipynb
evaluation/
  act_policy_adapter.py
  rollout_act.py
  collect_dagger_v006.py
  rollout_video_v006b.py
  compose_four_panel_v006b.py
docs/
```

The raw episodes, converted HDF5 files, trained checkpoints, and rendered video outputs are intentionally **not committed**. Their default local directories are `data/`, `training/checkpoints/`, and `evaluation/results/`.

## Dataset: Experiment 006-B

100 demonstration episodes, each **27 s at 50 Hz (1,350 samples)**, organized as follows:

| Group | Episodes | Conditions | Train | Validation |
|---|---:|---|---:|---:|
| A | 0001–0040 | Cube XY offset within ±5 mm | 32 | 8 |
| B | 0041–0070 | Cube XY offset within ±20 mm | 24 | 6 |
| C | 0071–0090 | Random joint perturbation and teacher recovery | 16 | 4 |
| D | 0091–0100 | ACT→teacher DAgger-style collection | 8 | 2 |
| **Total** | **100** | | **80** | **20** |

Group C consists of 8 single-joint ±0.01 rad runs, 6 single-joint ±0.03 rad runs, and 6 multi-joint runs with perturbations up to ±0.03 rad. Perturbations occur at randomized times in roughly 3–6 seconds.

The train/validation episode lists were generated with seed 42 and are recorded in `training/experiment_006b_split.json`. The 80/20 split is by **episode**, with no overlapping episodes.

### Generate teacher data

For example, generate the A and B groups in a fresh RAW directory:

```bash
python mujoco_generator/scripts/collect_demonstrations_v006b.py \
  --episodes 40 --range-mm 5 --seed 4001 --start-id 1 \
  --perturb-mode none --output-dir data/raw_v006b_final

python mujoco_generator/scripts/collect_demonstrations_v006b.py \
  --episodes 30 --range-mm 20 --seed 5001 --start-id 41 \
  --perturb-mode none --output-dir data/raw_v006b_final
```

Example Group C generation:

```bash
python mujoco_generator/scripts/collect_demonstrations_v006b.py \
  --episodes 8 --range-mm 10 --seed 6001 --start-id 71 \
  --perturb-mode single --perturb-rad 0.01 --output-dir data/raw_v006b_final

python mujoco_generator/scripts/collect_demonstrations_v006b.py \
  --episodes 6 --range-mm 10 --seed 6002 --start-id 79 \
  --perturb-mode single --perturb-rad 0.03 --output-dir data/raw_v006b_final

python mujoco_generator/scripts/collect_demonstrations_v006b.py \
  --episodes 6 --range-mm 10 --seed 6003 --start-id 85 \
  --perturb-mode multi --perturb-rad 0.03 --output-dir data/raw_v006b_final
```

**DAgger reproducibility limitation:** Group D was collected using a separately trained **Experiment 006-A ACT checkpoint** and `evaluation/collect_dagger_v006.py`. The 006-A checkpoint is not included in this repository. Reproducing the *exact* 100-episode dataset therefore requires supplying that checkpoint (with matching RMB model metadata) and the original DAgger collection parameters. The Group D RAW metadata was made compatible with the converter by adding the standardized fields `success`, `num_steps`, `image_count`, `camera_name`, and `image_shape`, while preserving the original provisional-success metadata. Do not treat DAgger geometric success as evidence of autonomous ACT success.

### Convert RAW episodes to RMB HDF5

```bash
python data_pipeline/converters/raw_to_rmb.py \
  --input-dir data/raw_v006b_final \
  --output-dir data/rmb/pick_and_place_v006b
```

The completed experiment uses **100 HDF5 episodes**. For exact split reproduction, use the filenames in the committed split JSON. The raw data and converted files themselves are not hosted here.

## ACT training

Training was run from scratch on a Colab CUDA GPU using `notebooks/train_act_colab_v006b.ipynb` and `training/scripts/train_act_v006b.py`.

| Parameter | Value |
|---|---|
| Training / validation episodes | 80 / 20 |
| Epochs | 100 |
| Batch size | 4 |
| Action chunk size | 25 |
| Skip | 2 |
| Seed | 42 |
| Observation | `measured_joint_pos` (7D), RGB `front` |
| Action | `command_joint_pos` (7D) |
| Normalization statistics | Computed on training episodes only |

The 7D action comprises six joint position targets (rad) and a gripper actuator command (0–255). `skip=2` on 50 Hz source data gives a 25 Hz action interval (0.04 seconds). A checkpoint and matching `model_meta_info.pkl` must be kept together for inference.

Example training CLI (with a prepared Python 3.12 CUDA environment):

```bash
python training/scripts/train_act_v006b.py \
  --dataset-dir data/rmb/pick_and_place_v006b \
  --split-json training/experiment_006b_split.json \
  --checkpoint-dir training/checkpoints/act_experiment_006b \
  --rmb-root app/RoboManipBaselines \
  --epochs 100 --batch-size 4 --chunk-size 25 --skip 2
```

This script requires exactly the 100 HDF5 files named in the split JSON and refuses a non-empty checkpoint output directory.

## Evaluation and four-panel video

The video compares three matched initial cube offsets:

1. Task 1: `(0, 0)` mm
2. Task 2: `(+10, +10)` mm
3. Task 3: `(-10, -10)` mm

For each task, collect three trajectories: `act`, `assisted` (teacher for the first 15 s, then ACT), and `teacher` (reference). The assisted mode preserves the simulation state when control switches. Example for Task 1:

```bash
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
CKPT=training/checkpoints/act_experiment_006b/policy_best.ckpt

python evaluation/rollout_video_v006b.py \
  --mode act --checkpoint "$CKPT" \
  --output-dir evaluation/results/video_test_act_only \
  --max-seconds 27 --cube-offset-mm 0 0 --replan-every 1 --fps 25

python evaluation/rollout_video_v006b.py \
  --mode assisted --checkpoint "$CKPT" \
  --output-dir evaluation/results/video_test_assisted15 \
  --switch-seconds 15 --max-seconds 27 \
  --cube-offset-mm 0 0 --replan-every 1 --fps 25

python evaluation/rollout_video_v006b.py \
  --mode teacher --output-dir evaluation/results/video_task1_teacher \
  --max-seconds 27 --cube-offset-mm 0 0 --fps 25
```

For Task 2, the output folders are `video_task2_act`, `video_task2_assisted`, and `video_task2_teacher`, with offset `10 10`. For Task 3, they are `video_task3_act`, `video_task3_assisted`, and `video_task3_teacher`, with offset `-10 -10`.

After all nine directories exist, render the four-panel demo:

```bash
python evaluation/compose_four_panel_v006b.py \
  --output evaluation/results/final_demo_v006b.mp4
```

The panels show (top-left) ACT Only, (top-right) Teacher → ACT, (bottom-left) six-joint RMSE versus the matching teacher trajectory, and (bottom-right) gripper commands. The rendered output was verified as **H.264, 1920×1080, 25 fps, 2,025 frames, 81 seconds**. The video file is an output artifact and is not committed in this code-only repository.

## Recorded results

| Test case | ACT-only final XY error | ACT-only provisional result | Teacher→ACT final XY error | Teacher→ACT provisional result |
|---|---:|---|---:|---|
| Task 1: `(0,0)` mm | 103.02 mm | Fail | 4.56 mm | Pass |
| Task 2: `(+10,+10)` mm | 142.13 mm | Fail | 4.66 mm | Pass |
| Task 3: `(-10,-10)` mm | 132.51 mm | Fail | 4.30 mm | Pass |

ACT-only maximum cube clearance was approximately **2.00 mm** for each recorded test. Teacher-assisted maximum clearance was approximately **98.8 mm**, including the teacher-controlled part of the trajectory.

**Interpretation:** Passing the *provisional geometric criterion* does not independently prove grasp quality or that ACT achieved the full task. The 15-second teacher segment already includes grasping and much of the transport; teacher-assisted results must not be presented as autonomous ACT task completion. The baseline ACT-only policy failed all three recorded cases, despite giving plausible gripper commands on some saved teacher states. This indicates a limitation in closed-loop execution that remains unresolved.

## Documentation and licensing

- [Environment setup](docs/01_manipulation_lab_setup.md)
- [MuJoCo teleoperation / data collection notes](docs/02_mujoco_teleop_data_collection.md)
- Third-party robot/gripper asset licenses are provided alongside the relevant files under `mujoco_generator/assets/`.
- RoboManipBaselines is included as a Git submodule; refer to its upstream repository for its own license and dependencies.

The project report will be added separately after repository cleanup.
