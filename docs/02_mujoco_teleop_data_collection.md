# Manipulation Lab — MuJoCo／テレオペ／RMBデータ収集・可視化手順

> **対象範囲：ACT学習に入る前まで（検証済み工程）**  
> 最終更新：2026-10-16  
> プロジェクト：Physical AI最終課題／将来のYUBI・ROS 2連携を見越した開発環境  
> ステータス：**MuJoCo起動〜関節角度の可視化まで実施済み**。ACT学習・ROS 2 Bridgeは本書の対象外。

## 1. 目的とシステム構成

現在のWSL2 Ubuntu上で **RoboManipBaselines + MuJoCo** を使用し、キーボードによるロボット操作からデータ記録・再生・解析までを行う。将来は同一GitHubのソースを **Ubuntu + NVIDIA RTX** に導入し、**YUBI-SW／ROS 2** と接続する。

```text
キーボード
    │
    ▼
RoboManipBaselines / Teleop.py
    │ Action
    ▼
MuJoCo (UR5e + Cable)
    │ Observation / RGB / Depth / Command
    ▼
RMB データセット（1エピソード＝1つの .rmb ディレクトリ）
    ├─ main.rmb.hdf5    : 時刻・観測値・指令値・報酬
    └─ *.rmb.mp4       : RGBおよびDepth動画
    │
    ├─ Teleop.py --replay_log : 記録した操作を再生
    └─ h5py + matplotlib    : データ構造確認とグラフ
```

**環境情報（今回のPC）**

| 項目 | 内容 |
|---|---|
| ホスト | Windows、AMD Radeon(TM) Graphics |
| Linux | WSL2 / Ubuntu 24.04 LTS / WSLg |
| Python | 3.12、プロジェクト専用 `.venv` |
| シミュレーション | MuJoCo、`MujocoUR5eCable` |
| RoboManipBaselines | Git submodule `app/RoboManipBaselines` |
| OpenGL | WSLgクラッシュ対策としてMesa `llvmpipe`（CPU描画） |
| PyTorch | 確認時 `2.10.0+cu128`、CUDA利用不可（今回は学習しない） |
| GitHub | `https://github.com/bashg0x/manipulation-lab`（Public） |
| ローカルのルート | `~/ai/research/manipulation_lab` |

> 本手順ではYUBIの既存環境 `~/ai/resarch/yubi_lab` を変更しない。WSLg表示とCUDAによる機械学習計算は別機能。

## 2. ソースコードの取得とディレクトリ構造

**新しいPCでGitHubリポジトリから再構築する場合**：

```bash
mkdir -p ~/ai/research
cd ~/ai/research
git clone --recurse-submodules https://github.com/bashg0x/manipulation-lab.git manipulation_lab
cd manipulation_lab
```

既にクローン済みでSubmoduleが未取得の場合：

```bash
cd ~/ai/research/manipulation_lab
git submodule update --init --recursive
```

構造（主な部分）：

```text
manipulation_lab/
├── .venv/                         # Git管理しない
├── app/
│   └── RoboManipBaselines/       # 外部OSS（Git submodule）
│       └── robo_manip_baselines/
│           ├── bin/Teleop.py
│           └── dataset/          # RMBの実際の保存先
├── ros2_ws/src/                   # 後続工程
├── src/adapters/                  # 後続工程
├── src/data/
├── src/evaluation/
├── scripts/
│   └── plot_joint.py             # 自作の解析スクリプト
├── docs/
├── configs/
├── datasets/                      # 別の保存先として将来利用予定
└── experiments/
```

> 過去に手動で `git clone --recursive ... app/RoboManipBaselines` した環境では、親リポジトリで **Git submoduleとして登録済み**であることを確認する。新環境では親リポジトリのコミットを使い、外部OSSを勝手に最新版へ更新しない。

## 3. Ubuntuの依存関係とPython仮想環境

今回の環境で使用したUbuntu 24.04の構成例：

```bash
sudo apt update
sudo apt install -y \
  python3-venv ffmpeg \
  libavcodec60 libavformat60 libavutil58 libavdevice60 \
  libavfilter9 libswscale7 libswresample4 \
  x11-apps mesa-utils

cd ~/ai/research/manipulation_lab
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cd app/RoboManipBaselines
python -m pip install -e .
python -m pip check
```

**既存のPCでは `.venv` を作り直す必要はない**。使用するときだけ：

```bash
cd ~/ai/research/manipulation_lab
source .venv/bin/activate
```

動作確認：

```bash
python -c 'import mujoco, h5py, matplotlib; print("Python packages OK")'
python -m pip check
```

TorchCodec／FFmpegの組合せによっては共有ライブラリ読み込みエラーが出ることがある。今回はFFmpeg関連のライブラリを導入して解消した。`pip` のエラーがあるときは、PyTorchを無条件に更新せず互換性を確認する。

## 4. WSLgのGUI／OpenGL確認（今回の重要なトラブル対策）

確認コマンド：

```bash
echo "$DISPLAY"
xeyes
# xeyesを終了してから
glxinfo -B
```

当初、MuJoCo起動時に次のエラーが発生した。

```text
GLFWError: X11: Failed to open display :0
mujoco.FatalError: gladLoadGL error
```

`xeyes` も起動できず、WSLgのログでXWaylandのクラッシュを確認した。AMDドライバー経由の描画を避けるため、**Windows側** `%USERPROFILE%\.wslgconfig` に以下を設定した。

```ini
[system-distro-env]
LIBGL_ALWAYS_SOFTWARE=1
```

変更後、PowerShellで：

```powershell
wsl --shutdown
```

Ubuntuを起動し直して `xeyes` が表示され、`glxinfo -B` に **`llvmpipe`** が表示されれば今回の対処は成功。llvmpipeはCPUによる描画なので、描画性能には制限がある。

**注意**：`/tmp/.X11-unix` の削除・アンマウント・シンボリックリンクの変更はしない。WSLg管理下のマウントであり、ほかのWSLアプリにも影響する可能性がある。

## 5. MuJoCo起動とキーボードテレオペ

```bash
cd ~/ai/research/manipulation_lab
source .venv/bin/activate
cd app/RoboManipBaselines/robo_manip_baselines

python ./bin/Teleop.py MujocoUR5eCable \
  --world_idx_list 0 5 \
  --input_device keyboard
```

確認済みの開始ログ例：

```text
[Teleop] Reset environment. demo_name: MujocoUR5eCable, world_idx: 0, episode_idx: 0
[Teleop] Press the 'n' key to proceed.
```

操作は**ウィンドウにフォーカスがあること**を確認する。操作開始手順は、画面案内に従って `n` → `n`（待機フェーズからテレオペへ）と進む。

| 操作 | キー |
|---|---|
| 手先X+／X− | `W`／`S` |
| 手先Y+／Y− | `A`／`D` |
| 手先Z+／Z− | `Q`／`E` |
| Roll−／Roll+ | `J`／`L` |
| Pitch+／Pitch− | `I`／`K` |
| Yaw+／Yaw− | `U`／`O` |
| グリッパを閉じる／開く | `Z`／`X` |
| 状態遷移・操作終了 | `N` |
| エピソードを保存 | `S`（終了後） |
| 保存せず破棄 | `F`（終了後） |
| アプリ終了 | `Esc` |

> `S` キーは移動の `S` と重複するが、**テレオペ実行中は移動、記録終了後のフェーズでは保存**という文脈で区別される。操作キーは制御座標系であり、画面上の上下左右と一致するとは限らない。

## 6. 1エピソードをRMB形式で保存

1. `n` → `n` でテレオペを開始。
2. キーボードでロボットを動かす。
3. `n` で記録フェーズを終了。
4. `s` でデータを保存。
5. 必要に応じて `Esc` で終了。

ファイル一覧を確認する：

```bash
cd ~/ai/research/manipulation_lab/app/RoboManipBaselines/robo_manip_baselines
find dataset -maxdepth 4 -type f | head -50
```

**今回保存できた実データ**：

```text
dataset/
└── MujocoUR5eCable_20261016_210135/
    └── MujocoUR5eCable_world0_000.rmb/
        ├── main.rmb.hdf5
        ├── front_rgb_image.rmb.mp4
        ├── front_depth_image.rmb.mp4
        ├── left_rgb_image.rmb.mp4
        ├── left_depth_image.rmb.mp4
        ├── right_rgb_image.rmb.mp4
        ├── right_depth_image.rmb.mp4
        ├── hand_rgb_image.rmb.mp4
        └── hand_depth_image.rmb.mp4
```

- `.rmb` は**ファイルではなく1エピソード分のディレクトリ**（RmbData-Compact形式）。
- `main.rmb.hdf5` に数値、時刻、指令、報酬が格納される。
- 4方向のRGBおよびDepth動画はMP4として別保存される。
- データセットディレクトリには自動生成された `.gitignore` もある。
- 新しい収集では日時・world番号・episode番号が変わる。必ず実ファイル名を確認する。

## 7. 保存したデータを再生

```bash
cd ~/ai/research/manipulation_lab/app/RoboManipBaselines/robo_manip_baselines

python ./bin/Teleop.py MujocoUR5eCable \
  --replay_log \
  ./dataset/MujocoUR5eCable_20261016_210135/MujocoUR5eCable_world0_000.rmb
```

画面案内に従って `n` で再生を進める。**今回のエピソードは再生成功を確認済み**。

> upstreamのドキュメントにあるエピソード名の例（`_env0_...`）と、今回の環境で実際に作られたファイル名（`_world0_...`）が異なる場合がある。`find` で確認した**実際のパス**を優先する。

## 8. HDF5の構造を確認

ターミナルでPython文を直接Bashに貼ると `Command 'from' not found` となる。以下のように `python - <<'PY'` を使用するか、`.py` ファイルとして保存して実行する。

```bash
cd ~/ai/research/manipulation_lab
source .venv/bin/activate

python - <<'PY'
from pathlib import Path
import h5py

path = (
    Path('app/RoboManipBaselines/robo_manip_baselines/dataset')
    / 'MujocoUR5eCable_20261016_210135'
    / 'MujocoUR5eCable_world0_000.rmb'
    / 'main.rmb.hdf5'
)

with h5py.File(path, 'r') as f:
    def show(name, obj):
        if isinstance(obj, h5py.Dataset):
            print(f'{name:44s} shape={str(obj.shape):12s} dtype={obj.dtype}')
    f.visititems(show)
PY
```

**今回確認したHDF5の全16データ項目（220ステップ）**：

| キー | 形状 | 備考 |
|---|---|---|
| `time` | `(220,)` | 時刻 [s] |
| `reward` | `(220,)` | 報酬 |
| `measured_joint_pos` | `(220, 7)` | 関節位置（6軸＋グリッパを含む構成） |
| `command_joint_pos` | `(220, 7)` | 関節位置指令 |
| `measured_joint_pos_rel` | `(220, 7)` | 相対関節位置 |
| `command_joint_pos_rel` | `(220, 7)` | 相対関節位置指令 |
| `measured_joint_vel` | `(220, 7)` | 関節速度 |
| `measured_eef_pose` | `(220, 7)` | 手先位置3＋四元数4 |
| `command_eef_pose` | `(220, 7)` | 手先位置姿勢の指令 |
| `measured_eef_pose_rel` | `(220, 6)` | 相対手先位置姿勢 |
| `command_eef_pose_rel` | `(220, 6)` | 相対手先位置姿勢指令 |
| `measured_eef_wrench` | `(220, 6)` | 手先力3＋トルク3 |
| `measured_gripper_joint_pos` | `(220, 1)` | グリッパ位置 |
| `command_gripper_joint_pos` | `(220, 1)` | グリッパ指令 |
| `measured_gripper_joint_pos_rel` | `(220, 1)` | 相対グリッパ位置 |
| `command_gripper_joint_pos_rel` | `(220, 1)` | 相対グリッパ指令 |

すべて `float64`。`measured_*` はMuJoCoのシミュレーション状態であり、実機センサ計測値ではない。`command_*` は学習時のAction候補だが、実際に使用するキーは学習の設定による。グリッパ列の単位が関節角度と同じとは限らない。

## 9. 関節角度（Command / Measured）を可視化

BashではなくPythonとして実行できるよう、親プロジェクトに `scripts/plot_joint.py` を配置する。

```bash
cd ~/ai/research/manipulation_lab
mkdir -p scripts
nano scripts/plot_joint.py
```

`plot_joint.py` の内容：

```python
from pathlib import Path

import h5py
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
HDF5_PATH = (
    PROJECT_ROOT
    / 'app/RoboManipBaselines/robo_manip_baselines/dataset'
    / 'MujocoUR5eCable_20261016_210135'
    / 'MujocoUR5eCable_world0_000.rmb'
    / 'main.rmb.hdf5'
)

with h5py.File(HDF5_PATH, 'r') as f:
    time = f['time'][:]
    measured = f['measured_joint_pos'][:]
    command = f['command_joint_pos'][:]

joint_idx = 0  # 最初のアーム関節
plt.figure(figsize=(10, 4))
plt.plot(time, measured[:, joint_idx], label='Measured')
plt.plot(time, command[:, joint_idx], '--', label='Command')
plt.xlabel('Time [s]')
plt.ylabel('Joint position [rad]')
plt.title(f'UR5e Joint {joint_idx}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```

実行：

```bash
cd ~/ai/research/manipulation_lab
source .venv/bin/activate
python scripts/plot_joint.py
```

**今回の確認結果**：横軸は約0〜7秒、Joint 0の指令値と実際の関節位置を2本の曲線で表示できた。指令値の段階的変化に対してMeasuredがわずかに遅れて追従する様子を観察した。これは制御の追従特性を示すが、グラフだけで制御器の内部方式は特定できない。

> 別エピソードを調べるときは `HDF5_PATH` を変更する。将来的にはCLI引数／YAMLでエピソードを選択できるように拡張する。

## 10. トラブルシューティング

| 症状 | 原因・対策 |
|---|---|
| `Command 'from' not found` | PythonコードをBashに貼り付けている。`python - <<'PY' ... PY` または `python path/to/file.py` を使う。 |
| `Could not load libtorchcodec`／`libavdevice.so.60` 不足 | UbuntuのFFmpeg共有ライブラリを確認：`ffmpeg -version`／`ldconfig -p \| grep libavdevice`。TorchCodecとTorchの互換性にも注意。 |
| `GLFWError: Failed to open display :0`／`gladLoadGL error` | まず `xeyes`、`glxinfo -B` を試す。WSLg障害の場合は前述の `.wslgconfig` を検討。 |
| キー操作が効かない | ウィンドウフォーカス、`n` でテレオペ段階に進んだかを確認。 |
| データが見つからない | `dataset/` は `app/RoboManipBaselines/robo_manip_baselines/dataset/`。`find` で日時付きディレクトリ名を確認。 |
| Windows再起動後にGUIが使えない | `xeyes`／`glxinfo -B` を再確認。WSLg設定を確認する。 |

### 公式資料

- [RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines)
- [Installation Guide](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/install.md)
- [Teleoperation](https://github.com/isri-aist/RoboManipBaselines/blob/master/robo_manip_baselines/teleop/README.md)
- [RMB Data Format](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/rmb_data_format.md)
- [Quick Start / ACT](https://github.com/isri-aist/RoboManipBaselines/blob/master/doc/quick_start.md)
