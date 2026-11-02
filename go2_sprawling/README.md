# Go2 Sprawling のロボットモデル

| ファイル | 内容 |
|---|---|
| `urdf/go2_orig.urdf` | 元の Unitree Go2 (Genesis 同梱の `genesis/assets/urdf/go2/urdf/go2.urdf` をそのまま複製) |
| `urdf/go2.urdf` | Go2 Sprawling: 付け根の関節の軸を鉛直にし、脚を胴体の横へ張り出して、後ろ脚を前脚の前後反転にした改造版 |
| `urdf/go2_strong.urdf` | Go2 Sprawling の関節のトルク上限を 2 倍、速度上限を 1.5 倍にした増強版 (宙返り用) |
| `dae/` | メッシュ (Genesis 同梱の Go2 のものをそのまま複製) |

元のモデルとメッシュは Genesis (Apache-2.0) に同梱の Unitree Go2 で、Unitree Robotics の
[unitree_ros](https://github.com/unitreerobotics/unitree_ros) (BSD-3-Clause) に由来する。
URDF の先頭のコメント (SolidWorks to URDF Exporter の記載) は元のファイルのまま残している。
