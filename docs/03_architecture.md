# Manipulation Lab — システムアーキテクチャ

## 1. 概要

本システムは、MuJoCoによるロボットシミュレーションとRoboManipBaselinesによる模倣学習を組み合わせ、Pick & Place動作の学習・評価を行う。

## 2. システム構成

```text
MuJoCo（UR5e + Robotiq）
    │
    ▼
教師制御・データ生成（Python）
    │ RAWデータ
    ▼
データ変換（Python）
    │ HDF5
    ▼
RoboManipBaselines
    │ ACTモデル学習
    ▼
学習済みモデル
    │
    ▼
MuJoCo（推論・評価）
    │
    ▼
結果分析・動画生成
```

## 3. 各コンポーネントの役割

| コンポーネント     | 役割                                 |
| ------------------ | ------------------------------------ |
| MuJoCo             | ロボット・物体の物理シミュレーション |
| 教師制御           | Pick & Placeの教師データ生成         |
| データ変換         | RAWデータを学習用HDF5へ変換          |
| RoboManipBaselines | ACTモデルの学習                      |
| 評価機能           | 学習モデルによる動作検証             |
| 動画生成           | 動作・誤差の可視化                   |

## 4. 開発環境

- Windows 11 / WSL2 / Ubuntu 24.04
- Python 3.12
- MuJoCo
- RoboManipBaselines
- Google Colab（GPU学習）
