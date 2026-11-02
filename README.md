# Go2 4脚ロボット強化学習環境 (Genesis + rsl_rl)

[Qiita: Genesisで4脚ロボットの強化学習](https://qiita.com/hEnka/items/cc5fd872eb0bf7cd3abc) の手順を、Ubuntu 24.04 / Python 3.12 / 現行 Genesis 向けに調整したもの。

## 動作環境

Ubuntu 24.04 / Python 3.12 / NVIDIA RTX 4070 (12 GB) で確認した。Genesis 1.4.5 (`setup.sh` でコミットを固定)、rsl-rl-lib 5.5.2、PyTorch 2.15.0。
資料の PDF 化 (`tools/build_report.py`) には Google Chrome または Chromium を使う。

学習済みのモデル (`logs/`) と動画 (`videos/`) はリポジトリに含めていない。各節の学習コマンドで作り直せる (評価・動画のコマンドは `logs/<実験名>/model_<N>.pt` を読む)。
`go2_sprawling/` のロボットモデルは Genesis 同梱の Unitree Go2 (unitree_ros 由来、BSD-3-Clause) を改造したもの ([go2_sprawling/README.md](go2_sprawling/README.md))。
資料の PDF はリポジトリに含めていない (`python tools/build_report.py` で Markdown から作る)。

ライセンス: MIT ([LICENSE](LICENSE))。ただし `go2_sprawling/` のロボットモデルとメッシュは元のライセンス (Apache-2.0 / BSD-3-Clause) に従う。

## セットアップ

```bash
./setup.sh
source .venv/bin/activate
```

## 学習・評価

examples 内のスクリプトは `from go2_env import Go2Env` で同じディレクトリを import し、`logs/` をカレントディレクトリに作るので、`Genesis/` から実行する。

```bash
cd Genesis
python examples/locomotion/go2_train.py                     # 学習 (4096並列環境, 101 iteration, GPU)
tensorboard --logdir logs                                   # 学習の監視 (任意)
python examples/locomotion/go2_eval.py -e go2-walking --ckpt 100   # ビューアで評価
```

主なオプション: `-e/--exp-name`（ログ名, `logs/<name>`）, `-b/--num-envs`, `--max-iterations`, `--seed`。
同じ実験名で学習し直すと `logs/<name>` は削除される。

## 全方向歩行 (前後・横移動・旋回)

`go2_omni/` は Genesis の `Go2Env` と設定を流用し、速度コマンドの範囲（前後 ±1.0 m/s、横 ±0.6 m/s、旋回 ±1.0 rad/s）と学習回数（1000 iteration）を上書きする。ログはプロジェクト直下の `logs/<exp>` に出る。プロジェクト直下から実行する。

```bash
python go2_omni/train.py                       # 学習 (RTX 4070 で約5分)
python go2_omni/eval.py --ckpt 999             # 固定コマンド (前進/後退/横/旋回など) ごとの追従誤差と転倒数
python go2_omni/eval.py --ckpt 999 --viewer    # ビューアでコマンドを切り替えながら表示 (Ctrl+C で終了)
```

コマンド範囲は `go2_omni/train.py` の `COMMAND_RANGES`、評価するコマンドは `go2_omni/eval.py` の `SCENARIOS` で変更する。

## 不整地歩行 (坂・凹凸・階段・障害物)

`go2_rough/` は `Go2Env` を元に書き直した独立の環境 `Go2RoughEnv`（`go2_rough/env.py`）を使う。プロジェクト直下から実行する。

```bash
python go2_rough/train.py                                   # 学習 (2000 iteration, RTX 4070 で約13分)
python go2_rough/eval.py --ckpt 1999                        # 地形タイプ x 難易度ごとの成功率と移動距離
python go2_rough/eval.py --ckpt 1999 --viewer --terrain stairs_up --level 7   # カメラ固定 (マウスでロボットを中心に回転・ズーム), --follow でロボットに付いて動く
python go2_rough/record.py --terrain stairs_up --level 7    # チェックポイントごとの比較動画 -> videos/
python go2_rough/record.py --tour                           # 最終モデルで各地形を巡る動画 -> videos/
python go2_rough/record.py -e go2-sprawling-rough-fast --story     # 学習の過程 (約 30 秒) + 最終モデルの見せ場 (約 60 秒) -> videos/go2-sprawling_rough-terrain_story_ckpt<N>.mp4
python go2_rough/record_crowd.py                             # 768 台を全マスに置き、チェックポイントごとに同じカメラの動きで撮って並べる (約 1 分) -> videos/go2-sprawling_rough-terrain_crowd_ckpt<N>.mp4
```

キーボード操縦（ビューアのウィンドウをアクティブにして操作。キーを押している間だけ速度指令が入る）:

```bash
python go2_rough/teleop.py                          # 上り階段・難易度4から出発。--terrain / --level で変更
```

歩行とジャンプを別のモデルに任せることもできる（`go2_rough/policy_switch.py`）。ふだんは `-e` の歩行モデルで動き、J キーを押すと速度指令を 0 にして立ち止まるのを待ち（止まった状態が 0.3 秒続くか最大 1.5 秒）、`--jump-exp` のモデルに切り替えて跳ぶ。着地から 0.5 秒後に歩行モデルに戻す。ジャンプ指令の観測を持たない歩行モデルには、その 2 項目を除いた観測を渡す。

```bash
python go2_rough/teleop.py -e go2-sprawling-rough-fast --ckpt 2999 --jump-exp go2-sprawling-jump-v3 --jump-ckpt 2900
```

| キー | 動作 |
|---|---|
| ↑ / ↓ | 前進 / 後退（最大 0.8 m/s, `--max-vx`） |
| ← / → | 左旋回 / 右旋回（最大 1.0 rad/s, `--max-wz`） |
| Q / E | 左 / 右へ横移動（最大 0.5 m/s, `--max-vy`） |
| J | その場ジャンプ（`--jump` で学習したモデルか `--jump-exp` を指定したとき。ジャンプ中は移動の指令を無視する） |
| Space | 停止 |
| X | 出発地点に戻す |
| 1 / 2 / 3 | 視点切替: 後方俯瞰（向きも追従） / 位置だけ追従 / 固定。2 と 3 はマウスのドラッグとホイールでロボットを中心に回転・ズームできる |
| ホイール / PageUp / PageDown | 後方俯瞰のズーム（見下ろす角度は保ったまま近づく / 離れる。距離は 0.3〜5 倍） |
| Esc | 終了 |

最初の視点は後方俯瞰（`--camera chase|follow|fixed` で変更、距離と高さは `--chase-distance` / `--chase-height`）。平地で学習したモデルでは地面を 50 cm 四方の市松模様にする（表示だけで物理には影響しない。起伏のある地形は段差が見にくくなるので無地。`--checker-size` でマスの大きさ [m] を指定、0 で無地）。W/A/S/D などはビューア既定の機能（I キーで一覧）なので使っていない。

平面版からの主な違い:

- 地形: 6 タイプ（上り坂/下り坂/凹凸/上り階段/下り階段/障害物）x 8 難易度のグリッド。各マス 6 m 四方で、中央の平坦部から出発する。最高難易度は傾斜 0.4、段差 13 cm、障害物 15 cm
- カリキュラム: マスの半分（3 m）以上進めたら難易度 +1、指令どおりなら進めたはずの距離の半分未満なら -1。最高難易度をクリアしたらランダムな難易度に戻る（そのため平均難易度は最大でも 3.5 前後）
- 観測: 胴体周囲 13 x 9 点の地面の高さスキャンを追加。critic には胴体の並進速度も渡す（`obs_groups`）
- 終了: 胴体の接地か 60° 以上の傾き（平面版の roll/pitch 10° は坂で使えない）。地形の端に着いた場合はタイムアウト扱い
- 報酬: 胴体の高さを足元の地面基準に変更し、足の滞空時間・太もも/股関節の衝突・胴体の揺れ（`ang_vel_xy`）・接地中の足先の滑り（`feet_slip`）・付け根（hip）の基準姿勢からの振れ（`hip_deviation`）を追加。滞空時間の目標は 0.25 s。合計報酬は 0 未満を切り捨てる（`only_positive_rewards`）
- Go2 の足先は calf リンクに統合されているため、接地判定は calf リンクの接触力で行う
- 速度指令のカリキュラム: 前後の速度は ±1.0 m/s から始め、追従報酬の 1 ステップあたりの平均が 0.8 を超えるたびに 0.1 m/s ずつ広げて ±2.0 m/s まで上げる（判定は 10 秒ごと。`command_cfg` の `lin_vel_x_init_range` / `curriculum_*`）。現在の上限は tensorboard の `max_command_x`。操縦で速く走らせるときは `teleop.py --max-vx 2.0`
- モーターの速度上限: Genesis は URDF の velocity を使わないので、`env_cfg["dof_vel_limits"]`（hip/thigh 30.1, calf 20.07 rad/s）を与えると、関節速度が上限の 70% を超えたところから回る向きのトルク上限を直線的に下げ、上限で 0 にする（`env.py` の `_apply_motor_speed_limits`。環境ごとにトルク上限を変えるため `batch_dofs_info` を使う）。キーのない古い `cfgs.pkl` では使われない
- その場ジャンプ（`train.py --jump`）: ジャンプ指令を受けると 2 秒間は速度指令が 0 になり、その間に跳んで着地すると完了。観測に「指令中（着地まで）」と「指令からの経過時間」の 2 つを追加する。報酬は、指令中に 4 本とも浮いている間のロボット全体の重心が放物線で達する最高点の上昇量（`jump_com`。空中で脚を縮めても増えないので、体ごと跳ぶ動きだけが報われる）、空中での基準姿勢からのずれの罰（`jump_flight_pose`。足を跳ね上げる動きを抑える）、指令中の胴体の上昇量（`jump_stretch`、小さめ）。指令中は胴体の高さ・上下速度・基準姿勢との差の罰を外すので、踏み切り前にしゃがむなどの準備動作は自由。学習中は平均 6 秒に 1 回ランダムに指令を出す（`command_cfg["jump"]`）。ジャンプの報酬が強いので長く学習すると歩行を忘れる。途中のチェックポイントを `eval.py --scenarios` で確かめて選ぶ。評価は `eval.py --ckpt N --jump`（成功率・重心と胴体の上昇量・足先の高さ・滞空時間・水平のずれ・転倒数）
- 対称性: rsl_rl の対称性拡張（PPO の `symmetry_cfg`）で、学習データに左右・前後・180° 回転の鏡映を加えて 4 倍にし、どちら向きにも同じ歩き方をするよう誘導する（`go2_rough/symmetry.py`。1 iteration が約 2.5 倍遅くなる）。`--no-symmetry` で無効にできる。鏡映では脚を入れ替え、付け根（hip）の角度だけ符号を反転する。Go2 Sprawling の脚の配置に合わせた対応なので、URDF の脚の付き方を変えたら見直す
- ロボットのモデルは Go2 Sprawling（`go2_sprawling/urdf/go2.urdf`。Genesis 同梱の Go2 の脚の付き方を変えた改造版。変更前は `go2_sprawling/urdf/go2_orig.urdf`。`train.py` の `env_cfg["urdf_file"]`）。変更後は `-e` で別の実験名を付けて学習し直す（同じ実験名だと `logs/<exp>` が削除される）

## 後方宙返り (バックフリップ)

`train.py --flip` で宙返りだけを学習する（`go2_rough/flip.py`、平地）。各エピソードで立った状態から 1 回だけ宙返りの指令を出し、指令からの経過時間 t で区切った報酬を与える（[ziyanx02/Genesis-backflip](https://github.com/ziyanx02/Genesis-backflip) の考え方に合わせている）。

- t < 0.5 s（構え）: 足を地面から離さない。0.5〜0.75 s（踏み切り）: 上向きの速度に報酬。0.5〜1.0 s（回転）: 後ろ向きの回転速度に報酬（12 rad/s まで）、胴体の向きの目標を 0 から 360° まで回す。1.4 s 以降: 立ったときの高さに戻す
- 常に: 横倒れ・ひねり・左右非対称な行動・胴体の接地に罰
- 宙返りの指令中は傾きで終了しない。学習中は失敗しても時間切れまで続け、報酬の合計の 0 での切り捨ても外す（罰が中心の報酬なので）
- 前後の鏡映を含む対称性の誘導は使わない
- 観測はジャンプのモデルと同じ並びなので、teleop の `--jump-exp` に指定すると J キーで宙返りする

元の Go2 のトルク上限では回転が足りない（踏み切りで太もも・付け根が 23.7 Nm に張り付き、後ろ向きの回転は約 6 rad/s で、後ろ足で着地して戻る）。`--strong` で、関節のトルク上限を 2 倍・速度上限を 1.5 倍にした増強版 `go2_sprawling/urdf/go2_strong.urdf`（質量・形状は同じ）を使う。

```bash
python go2_rough/train.py -e go2-sprawling-strong-flip --flip --strong          # 学習
python go2_rough/eval.py -e go2-sprawling-strong-flip --ckpt N --flip          # 回転数・成功率・胴体の接地
python go2_rough/eval.py -e go2-sprawling-strong-flip --ckpt N --viewer --follow   # 宙返りを繰り返し表示
python go2_rough/record_flip.py -e go2-sprawling-strong-flip-v3 --ckpt 2999       # 4 視点 (真横・正面・斜め後ろ上・追従) の動画。等速 + 0.25 倍スロー -> videos/go2-sprawling-strong_backflip_ckpt<N>.mp4
python go2_rough/record_crowd_flip.py                        # 400 台を格子に並べて一斉に宙返り。3 段階の学習を通しの iteration で並べ、最後に等速とスロー -> videos/go2-sprawling-strong_backflip_crowd_ckpt<N>.mp4
python go2_rough/teleop.py -e go2-sprawling-rough-fast --ckpt 2999 --jump-exp go2-sprawling-strong-flip-v3 --jump-ckpt 2600 --terrain rough --level 0
```

teleop で宙返りのモデルを指定すると、歩行のモデルも増強版のロボットで動かす。宙返りは平地でしか学習していない。

## 環境の中身

- `Genesis/examples/locomotion/go2_env.py` — `Go2Env`: 並列シミュレーション、PD 制御 (kp=20, kd=0.5)、観測、報酬、リセット
- `Genesis/examples/locomotion/go2_train.py` — `get_cfgs()` に環境・観測・報酬・速度コマンドの設定、`get_train_cfg()` に PPO の設定
  - 行動: 12関節の目標角度オフセット (`action_scale` 0.25)
  - 報酬: 速度追従 (lin/ang)、z 方向速度、胴体高さ (目標 0.3m)、行動変化率、デフォルト姿勢との差
  - 速度コマンド: デフォルトは前進 0.5 m/s 固定 (`command_cfg` の範囲を広げると全方向歩行になる)
  - 終了条件: roll/pitch が 10° を超えたとき
- 設定は学習時に `logs/<exp>/cfgs.pkl` に保存され、評価時はそれを読み込む

## 関節テスト (tools/joint_test.py)

URDF の各関節を GUI のスライダーで動かして確認する。スライダーは PD 制御の目標角で、学習と同じ kp/kd・URDF のトルク上限で関節が追従する。
「sweep」をオンにした関節は可動域内を往復し続ける。各リンクの座標軸は左パネルの Visualization -> Link Frame で表示できる。
GUI に `imgui-bundle` が必要（`uv pip install "imgui-bundle>=1.93.0"`）。

```bash
python tools/joint_test.py                                    # go2_sprawling/urdf/go2.urdf を空中に固定
python tools/joint_test.py --free                             # 地面に立たせる
python tools/joint_test.py --urdf go2_sprawling/urdf/go2_orig.urdf   # 変更前の URDF と比べる
```

## 改造の解説動画 (tools/record_sprawling_explainer.py)

元の Go2 と Go2 Sprawling を左右に並べ、改造の 3 点（付け根の関節の軸・脚の取り付け位置と向き・後ろ脚の前後反転）を関節の軸の矢印つきで見せたあと、学習済みの方策で歩く様子を並べる（約 67 秒、日本語の字幕）。

```bash
python tools/record_sprawling_explainer.py   # -> videos/go2-sprawling_explainer.mp4
```

## 資料 (docs/)

`docs/go2_sprawling_report.md`（改造・各タスクの学習・工夫をまとめた A4 2 ページの資料）を HTML にして、Chrome で PDF に印刷する。見た目は `docs/report.css`、図は `docs/img/`。

```bash
python tools/build_report.py   # -> docs/go2_sprawling_report.html / .pdf
```

Markdown では、画像だけの段落は説明がキャプションの図になり、`\newpage` だけの行で改ページ、段落の次の行の `{: .note }` でクラスが付く。横に並べるときは `<div class="row" markdown="1">` の中に `<div class="col" markdown="1">` を置く。

## ジャンプの確認 (tools/jump_test.py)

強化学習なしの決め打ち動作（しゃがむ → 目標角を一気に脚を伸ばした姿勢へ → 基準姿勢に戻す）で、ロボットが跳べるかを確かめる。
胴体の上昇量・足先の最高高さ・滞空時間・関節速度とトルクの最大を表示する。Genesis は URDF の velocity 上限を使わないので、関節速度が上限を超えていないかは表示で確認する。

```bash
python tools/jump_test.py                     # 学習と同じ kp=20
python tools/jump_test.py --kp 40 --video     # 横から見た動画 (0.5 倍速) -> videos/jump_kp40.mp4
python tools/jump_test.py --extend 0.5 -0.84  # 伸ばした姿勢 (太もも, すね) [rad]。--crouch でしゃがんだ姿勢
```
