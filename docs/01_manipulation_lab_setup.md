# Manipulation Lab — 開発環境構築・起動手順

## 1. 目的・構成

Physical AI最終課題向けに、**RoboManipBaselines + MuJoCo** でUR5eのテレオペ、データ収集、将来的な模倣学習を実施する。次の段階で **ROS 2 Jazzy** を状態・操作指令の共通インターフェースとして加え、将来の **YUBI-SW / YUBIデバイス** との接続に備える。

- ホスト：Windows + WSL2（Ubuntu 24.04、WSLg）
- Python：3.12（プロジェクト専用 `.venv`）
- シミュレータ：MuJoCo
- ロボット：UR5e（`MujocoUR5eCable`）
- AI：RoboManipBaselines（ACTを第一候補、学習は後続工程）
- 作業ディレクトリ：`~/ai/research/manipulation_lab`

## 2. ディレクトリ作成・取得

```bash
mkdir -p ~/ai/research/manipulation_lab
cd ~/ai/research/manipulation_lab
mkdir -p app ros2_ws/src src/adapters src/data src/evaluation configs datasets experiments docs

git clone --recursive https://github.com/isri-aist/RoboManipBaselines.git app/RoboManipBaselines
```

配置イメージ：

```text
manipulation_lab/
├── app/RoboManipBaselines/  # 外部OSS
├── ros2_ws/src/             # 今後のROS 2 Bridge
├── src/{adapters,data,evaluation}/
├── configs/
├── datasets/                # 原則Git管理しない
├── experiments/             # 大容量成果物はGit管理しない
└── docs/
```

## 3. Python環境と依存ライブラリ

```bash
cd ~/ai/research/manipulation_lab
sudo apt update
sudo apt install -y python3-venv ffmpeg \
  libavcodec60 libavformat60 libavutil58 libavdevice60 \
  libavfilter9 libswscale7 libswresample4 \
  x11-apps mesa-utils

python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

cd app/RoboManipBaselines
python -m pip install -e .
python -m pip check
python -c "import mujoco, torchcodec; print('MuJoCo:', mujoco.__version__, '/ TorchCodec OK')"
```

> Ubuntu 24.04のFFmpeg 6系列向けのパッケージ名。PyTorchとTorchCodecの組み合わせはインストール時点のバージョンにより確認すること。ACT用の追加依存関係は学習段階で導入する。

## 4. WSLg（GUI / OpenGL）確認

```bash
echo "$DISPLAY"                    # 通常 :0
ls -l /tmp/.X11-unix/X0            # X11ソケット
xeyes                              # Windows側にウィンドウが出るか
# xeyesを閉じたあと
glxinfo -B                         # OpenGL renderer を確認
```

このPCではWSLgのXWaylandがAMDドライバー経由でクラッシュしたため、**Windows側** の `%USERPROFILE%\.wslgconfig` に次を設定して復旧した。

```ini
[system-distro-env]
LIBGL_ALWAYS_SOFTWARE=1
```

変更後、PowerShellで `wsl --shutdown` を実行し、Ubuntuを再起動する（事前にWSL上の作業・コンテナを停止）。この回避策では `glxinfo -B` に `llvmpipe` が表示され、CPUによる描画となるため描画性能が低下する可能性がある。

## 5. テレオペ起動

```bash
cd ~/ai/research/manipulation_lab
source .venv/bin/activate
cd app/RoboManipBaselines/robo_manip_baselines

python ./bin/Teleop.py MujocoUR5eCable \
  --world_idx_list 0 5 \
  --input_device keyboard
```

起動例：

```text
[Teleop] Reset environment. demo_name: MujocoUR5eCable, world_idx: 0, episode_idx: 0
[Teleop] Press the 'n' key to proceed.
```

画像ウィンドウにフォーカスして `n` で準備→もう一度 `n` でテレオペ開始。終了時は `n` → `s`（保存）または `f`（破棄）。`Esc` でプログラム終了。

**操作キー（`KeyboardInputDevice.py` に基づく）**

| 操作 | キー |
|---|---|
| X± | W / S |
| Y± | A / D |
| Z± | Q / E |
| Roll ± | L / J |
| Pitch ± | I / K |
| Yaw ± | U / O |
| グリッパを閉じる / 開く | Z / X |

ロボットの制御座標系に対する移動で、画面の上下左右とは限らない。キーを短く押して動作を確かめる。

## 6. トラブルシューティング

| 症状 | 確認・対処 |
|---|---|
| `Could not load libtorchcodec` | `ffmpeg -version`、`ldconfig -p \| grep libavdevice` を確認し、上記FFmpeg共有ライブラリを導入。`python -m pip show torch torchcodec` でバージョン互換性を確認。無条件にPyTorchを再インストールしない。 |
| `GLFWError: Failed to open display :0` / `gladLoadGL error` | `xeyes` と `glxinfo -B` を先に確認。GUIそのものが失敗する場合はMuJoCoではなくWSLgを調査。 |
| `xeyes` / `glxinfo` が `:0` に接続できない | `/tmp/.X11-unix/X0`、`/mnt/wslg/weston.log`、`/mnt/wslg/stderr.log` を確認。`xserver crashing too fast` や `amdxc64.so` を含むクラッシュなら上記ソフトウェア描画を試す。 |
| `/tmp/.X11-unix` を削除できない | WSLgが読み取り専用でマウントしていることがある。**削除・アンマウント・リンク変更をしない**。`findmnt /tmp/.X11-unix` で確認。 |
| `n` や操作キーが反応しない | RoboManipBaselinesの画像ウィンドウにフォーカス。テレオペ開始フェーズまで `n` で進んでいるか確認。 |
| 描画が遅い | `llvmpipe` はCPU描画。まず動作確認を優先し、必要に応じてWindows側AMDドライバー・WSLgの更新を検討。 |

## 7. GitHub管理時の注意

1. `app/RoboManipBaselines` は**外部のGitリポジトリ**。自作コードと混ぜず、最終課題リポジトリでは **Git submodule** として登録するか、取得手順・コミットIDだけを記録する。既存の内部 `.git` を削除しない。
2. `.venv/`、`datasets/`、`experiments/` の大容量成果物、ログ、動画、秘密情報、個人情報はGitに含めない。必要な小さなサンプルデータだけ選択して公開する。
3. 動作確認後に `git -C app/RoboManipBaselines rev-parse HEAD` と `python -m pip freeze` を控え、環境再現性を確保する。
4. 提出用GitHubは **Public** が必須。YUBI-SWの業務用データ・非公開コードは含めない。

## 8. 次の作業

- [x] RoboManipBaselines / MuJoCoの起動
- [x] WSLg描画の復旧（llvmpipe）
- [ ] UR5eのキーボード操作確認
- [ ] 1エピソードの保存・再生とデータ形式の確認
- [ ] ACTで模倣学習・推論
- [ ] ROS 2 Bridgeを追加し、将来YUBI-SW入力に差し替え可能なAdapterを実装
- [ ] 独自タスク・動画・PDFレポートを作成

参考：[RoboManipBaselines](https://github.com/isri-aist/RoboManipBaselines)、[KeyboardInputDevice.py](https://github.com/isri-aist/RoboManipBaselines/blob/master/robo_manip_baselines/teleop/KeyboardInputDevice.py)
