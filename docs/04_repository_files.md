# Manipulation Lab — GitHub登録ファイル一覧

## 1. 概要

本書は、GitHubリポジトリ `manipulation-lab` に登録している主要ファイルの役割を簡単に説明する。対象は **Experiment 006-B（MuJoCoによるPick & Placeの模倣学習）** とする。

## 2. 処理の流れ

```text
MuJoCo環境・教師制御
  → 教師データ生成（RAW）
  → RMB形式へ変換（HDF5）
  → データ分割（80/20）・ACT学習
  → MuJoCoで推論・評価
  → 比較動画作成
```

## 3. ファイル構成と役割

| パス | 役割 |
|---|---|
| `README.md` | プロジェクト全体の概要、実験条件、実行手順、結果 |
| `.gitignore` | データ、モデル、録画結果、仮想環境などをGit管理から除外 |
| `.gitmodules` | RoboManipBaselinesサブモジュールの登録情報 |
| `app/RoboManipBaselines/` | ACT学習処理を提供する外部フレームワーク（Gitサブモジュール） |
| `mujoco_generator/environments/pick_and_place_camera.xml` | UR5e、グリッパー、キューブ、カメラ等を定義したMuJoCo環境 |
| `mujoco_generator/assets/robots/ur5e/assets/*.obj` | UR5eの3D形状データ |
| `mujoco_generator/assets/grippers/robotiq_2f85/assets/*.stl` | Robotiqグリッパーの3D形状データ |
| `mujoco_generator/assets/robots/ur5e/{LICENSE,README.md}` | UR5eモデルのライセンス・出典情報 |
| `mujoco_generator/assets/grippers/robotiq_2f85/{LICENSE,README.md}` | グリッパーモデルのライセンス・出典情報 |
| `mujoco_generator/controllers/pick_place_controller.py` | Pick & Place動作の基本となる教師制御 |
| `mujoco_generator/controllers/pick_place_controller_v005.py` | 今回の実験で用いた教師制御のV005版 |
| `mujoco_generator/controllers/pose_ik_controller.py` | 手先の目標位置・姿勢を関節指令へ変換する逆運動学（IK） |
| `mujoco_generator/collectors/episode_collector.py` | 1エピソード分の状態・指令・画像等の記録 |
| `mujoco_generator/scripts/collect_demonstrations_v006b.py` | 物体位置の変更や関節外乱を含む教師データを生成 |
| `data_pipeline/converters/raw_to_rmb.py` | RAWエピソードと画像をRoboManipBaselines用HDF5に変換・検証 |
| `training/experiment_006b_split.json` | 100エピソードの学習80件・検証20件の固定割り当て |
| `training/scripts/train_act_v006b.py` | RMBの学習処理を呼び出す006-B専用ACT学習設定・実行スクリプト |
| `notebooks/train_act_colab_v006b.ipynb` | Google Colab（GPU）での学習手順 |
| `evaluation/act_policy_adapter.py` | 学習済みACTモデルを推論時に呼び出すための接続処理 |
| `evaluation/rollout_act.py` | MuJoCoでのACT評価に共通するモデル設定・状態初期化等の処理 |
| `evaluation/collect_dagger_v006.py` | ACTが到達した状態から教師制御によるデータを収集 |
| `evaluation/rollout_video_v006b.py` | ACT単独・教師単独・教師→ACTの動作と評価用データを記録 |
| `evaluation/compose_four_panel_v006b.py` | 動作映像と関節誤差・グリッパー指令を4分割動画に合成 |
| `docs/01_manipulation_lab_setup.md` | 開発環境の構築・起動方法 |
| `docs/02_mujoco_teleop_data_collection.md` | MuJoCoテレオペ、データ保存・再生・可視化手順 |

`docs/03_architecture.md` はシステム構成の説明文書として別途作成する想定であり、GitHubに登録済みかどうかは本書の対象外とする。

## 4. GitHubに保存しないファイル

| 保存先 | 内容 |
|---|---|
| `data/` | 生成したRAWデータ・変換後HDF5・画像 |
| `training/checkpoints/` | 学習済みモデルとチェックポイント |
| `evaluation/results/` | 推論結果、比較用画像、動画など |
| `.venv/` | Python仮想環境 |

これらは容量が大きいため、原則ローカルで管理する。なお、006-BのDAggerデータを**完全に再生成**するには、GitHubに含めていない006-Aの学習済みモデルなどが別途必要となる。
