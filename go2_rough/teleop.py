"""学習済みの不整地歩行方策をキーボードで操縦する。

ビューアのウィンドウをアクティブにして操作する。キーを押している間だけその方向の速度指令が入る。
  ↑ / ↓   前進 / 後退
  ← / →   左旋回 / 右旋回
  Q / E    左へ横移動 / 右へ横移動
  J        その場ジャンプ (--jump で学習したモデルか、--jump-exp でジャンプ用のモデルを指定したとき。ジャンプ中は移動の指令を無視する)
  Space    停止 (押されているキーを全部解除)
  X        出発地点に戻す
  1 / 2 / 3  視点: 後方俯瞰 (向きも追従) / 位置だけ追従 / 固定。2 と 3 はマウスでロボットを中心に回転・ズームできる
  ホイール / PageUp / PageDown  後方俯瞰のズーム (見下ろす角度は保ったまま近づく / 離れる)
  Esc      終了
W/A/S/D などはビューアの既定機能 (I キーで一覧) に使われているので割り当てていない。
"""

import argparse
import math
import os
import pickle
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs
import genesis.utils.geom as gu
from genesis.vis.keybindings import Key, KeyAction, Keybind
from genesis.vis.viewer_plugins import EVENT_HANDLED, ViewerPlugin

from env import TERRAIN_TYPES
from eval import FixedCommandEnv
from policy_switch import PolicySwitcher, load_policy
from viewer_camera import RobotOrbitCamera

# キー -> 押している間に加える指令 (前後 m/s, 横 m/s, 旋回 rad/s) の向き
KEY_DIRECTIONS = {
    Key.UP: (1.0, 0.0, 0.0),
    Key.DOWN: (-1.0, 0.0, 0.0),
    Key.Q: (0.0, 1.0, 0.0),
    Key.E: (0.0, -1.0, 0.0),
    Key.LEFT: (0.0, 0.0, 1.0),
    Key.RIGHT: (0.0, 0.0, -1.0),
}
CAMERA_MODES = {Key._1: "chase", Key._2: "follow", Key._3: "fixed"}
# 位置だけ追従: 切り替えたときに、ロボット位置にこのオフセットを足した位置にカメラを置く
FOLLOW_OFFSET = np.array([2.0, 0.0, 2.5])
WORLD_UP = np.array([0.0, 0.0, 1.0])
# 後方俯瞰のズーム: 1 段階あたりの距離の倍率と、距離の倍率の範囲
ZOOM_STEP = 0.9
ZOOM_RANGE = (0.3, 5.0)


def zoom_chase(state, n_steps):
    """後方俯瞰のカメラ距離を n_steps 段階だけ縮める (負なら離れる)。"""
    state["chase_zoom"] = float(np.clip(state["chase_zoom"] * ZOOM_STEP**n_steps, *ZOOM_RANGE))


class ChaseZoomPlugin(ViewerPlugin):
    """後方俯瞰のときだけマウスホイールをカメラ距離の変更に使う。

    後方俯瞰ではカメラを毎フレーム上書きするため、ビューア標準のホイールズームはすぐ打ち消される。
    EVENT_HANDLED を返して標準のズームを止め、代わりに距離の倍率を変える。他の視点では標準のズームに任せる。
    """

    def __init__(self, state):
        super().__init__()
        self.state = state

    def on_mouse_scroll(self, x, y, dx, dy):
        if self.state["camera_mode"] != "chase":
            return None
        zoom_chase(self.state, dy)
        return EVENT_HANDLED


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-rough")
    parser.add_argument("--ckpt", type=int, default=1999)
    parser.add_argument("--terrain", choices=TERRAIN_TYPES, default="stairs_up", help="出発する地形")
    parser.add_argument("--level", type=int, default=4, help="出発する地形の難易度")
    parser.add_argument("--max-vx", type=float, default=0.8)
    parser.add_argument("--max-vy", type=float, default=0.5)
    parser.add_argument("--max-wz", type=float, default=1.0)
    parser.add_argument("--camera", choices=list(CAMERA_MODES.values()), default="chase", help="最初の視点")
    parser.add_argument("--chase-distance", type=float, default=1.8, help="後方俯瞰: ロボットの何 m 後ろか")
    parser.add_argument("--chase-height", type=float, default=1.2, help="後方俯瞰: ロボットの何 m 上か")
    parser.add_argument(
        "--jump-exp", type=str, default=None,
        help="ジャンプだけ別のモデル (train.py --jump) に任せる: J キーで立ち止まってからこのモデルに切り替え、着地後に戻す",
    )
    parser.add_argument("--jump-ckpt", type=int, default=None, help="--jump-exp のチェックポイント")
    parser.add_argument(
        "--checker-size", type=float, default=None,
        help="地面の市松模様の 1 マスの大きさ [m] (0 で無地)。省略時は平地なら 0.5 m、起伏のある地形なら無地 (段差が見にくくなるため)",
    )
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, _ = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    terrain_cfg["curriculum"] = False
    command_cfg["curriculum"] = False
    env_cfg["episode_length_s"] = 1e6
    if args.checker_size is None:
        terrain_cfg["checker_size"] = 0.5 if terrain_cfg.get("flat", False) else 0.0
    else:
        terrain_cfg["checker_size"] = args.checker_size
    if args.jump_exp:
        with open(os.path.join(ROOT, "logs", args.jump_exp, "cfgs.pkl"), "rb") as f:
            jump_env_cfg, _, _, jump_command_cfg, _, _ = pickle.load(f)
        # 環境はジャンプ用のモデルの学習と同じ条件 (ジャンプ指令の観測、ロボットのモデル、モーターの速度上限) にする。
        # 歩行のモデルも同じロボットで動かす (増強版でも質量・形状は同じで、トルクと速度の上限が大きいだけ)
        command_cfg["jump"] = jump_command_cfg["jump"]
        for key in ("urdf_file", "dof_vel_limits"):
            if key in jump_env_cfg:
                env_cfg[key] = jump_env_cfg[key]

    gs.init(backend=gs.gpu, logging_level="warning")
    env = FixedCommandEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=True,
    )
    env.terrain_types[:] = TERRAIN_TYPES.index(args.terrain)
    env.terrain_levels[:] = args.level

    policy = load_policy(env, args.exp_name, args.ckpt)
    switcher = None
    if args.jump_exp:
        switcher = PolicySwitcher(env, policy, load_policy(env, args.jump_exp, args.jump_ckpt))
        policy = switcher.act
    obs = env.reset()

    viewer = env.scene.viewer
    orbit = RobotOrbitCamera(viewer)
    pressed_keys = set()
    state = {"is_running": True, "is_reset_requested": False, "is_jump_requested": False, "camera_mode": args.camera, "chase_zoom": 1.0}
    # 後方俯瞰の注視点とカメラの向き (急な旋回やリセットで画面が揺れないよう平滑化する)
    chase = {"lookat": None, "yaw": 0.0}

    def look_at(pos, lookat):
        # viewer.set_camera_pose(pos=, lookat=) は前回のカメラの上方向を使い回すので、見下ろす姿勢で更新を続けると
        # 画面が傾いていく。世界の z 軸を上方向にした姿勢行列を渡し、画面の上を常に重力の逆向きにする
        viewer.set_camera_pose(
            pose=gu.pos_lookat_up_to_T(np.asarray(pos, dtype=np.float64), np.asarray(lookat, dtype=np.float64), WORLD_UP)
        )

    def apply_camera_mode():
        base = env.base_pos[0].cpu().numpy()
        chase["lookat"] = None
        orbit.reset()
        if state["camera_mode"] == "follow":
            # ロボットを斜め上から見る位置に置き、以後はメインループでロボットが動いた分だけ平行移動する
            look_at(base + FOLLOW_OFFSET, base)

    def update_chase_camera():
        base = env.base_pos[0].cpu().numpy()
        yaw = math.radians(env.base_euler[0, 2].item())
        if chase["lookat"] is None:
            chase["lookat"], chase["yaw"] = base, yaw
        # 角度の差を [-pi, pi] に折り返してから少しずつ近づける
        chase["yaw"] += 0.05 * math.remainder(yaw - chase["yaw"], 2 * math.pi)
        chase["lookat"] = 0.8 * chase["lookat"] + 0.2 * base
        heading = np.array([math.cos(chase["yaw"]), math.sin(chase["yaw"]), 0.0])
        zoom = state["chase_zoom"]
        pos = chase["lookat"] - zoom * args.chase_distance * heading + np.array([0.0, 0.0, zoom * args.chase_height])
        look_at(pos, chase["lookat"] + 0.3 * heading)

    def set_camera_mode(mode):
        state["camera_mode"], state["is_camera_changed"] = mode, True

    apply_camera_mode()
    state["is_camera_changed"] = False

    def on_press(key):
        pressed_keys.add(key)

    def on_release(key):
        pressed_keys.discard(key)

    keybinds = []
    for key in KEY_DIRECTIONS:
        keybinds.append(Keybind(f"teleop_press_{key.name}", key, KeyAction.PRESS, callback=on_press, args=(key,)))
        keybinds.append(Keybind(f"teleop_release_{key.name}", key, KeyAction.RELEASE, callback=on_release, args=(key,)))
    keybinds += [
        Keybind("teleop_stop", Key.SPACE, KeyAction.PRESS, callback=pressed_keys.clear),
        Keybind("teleop_jump", Key.J, KeyAction.PRESS, callback=state.__setitem__, args=("is_jump_requested", True)),
        Keybind("teleop_reset", Key.X, KeyAction.PRESS, callback=state.__setitem__, args=("is_reset_requested", True)),
        Keybind("teleop_quit", Key.ESCAPE, KeyAction.PRESS, callback=state.__setitem__, args=("is_running", False)),
    ]
    for key, mode in CAMERA_MODES.items():
        keybinds.append(Keybind(f"teleop_camera_{mode}", key, KeyAction.PRESS, callback=set_camera_mode, args=(mode,)))
    keybinds += [
        Keybind("teleop_zoom_in", Key.PAGEUP, KeyAction.HOLD, callback=zoom_chase, args=(state, 0.2)),
        Keybind("teleop_zoom_out", Key.PAGEDOWN, KeyAction.HOLD, callback=zoom_chase, args=(state, -0.2)),
    ]
    viewer.add_plugin(ChaseZoomPlugin(state))
    viewer.register_keybinds(*keybinds, overwrite=True)
    print(__doc__)

    max_command = torch.tensor([args.max_vx, args.max_vy, args.max_wz], dtype=gs.tc_float, device=gs.device)
    last_command = None
    with torch.no_grad():
        try:
            while state["is_running"]:
                if state["is_reset_requested"]:
                    state["is_reset_requested"] = False
                    obs = env.reset()
                    if switcher is not None:
                        switcher.reset()
                    state["is_camera_changed"] = True
                # ビューアの操作はキー入力のコールバック内ではなくメインループで行う
                if state["is_camera_changed"]:
                    state["is_camera_changed"] = False
                    apply_camera_mode()

                if state["is_jump_requested"]:
                    state["is_jump_requested"] = False
                    if switcher is not None:
                        switcher.request_jump(torch.zeros(1, dtype=gs.tc_int, device=gs.device))
                    elif env.jump_cfg:
                        env.trigger_jump(torch.zeros(1, dtype=gs.tc_int, device=gs.device))
                    else:
                        print("\nこのモデルはジャンプを学習していません (train.py --jump)")

                direction = [sum(KEY_DIRECTIONS[k][i] for k in pressed_keys) for i in range(3)]
                command = torch.tensor(direction, dtype=gs.tc_float, device=gs.device).clamp(-1.0, 1.0) * max_command
                # ジャンプ指令中は環境側で速度指令が 0 になるので、観測も合わせる。切り替え前の待機中も止まらせる
                if env.jump_timer[0] >= 0 or (switcher is not None and switcher.is_busy[0]):
                    command[:] = 0.0
                env.commands[:] = command
                FixedCommandEnv.fixed_command = tuple(command.tolist())
                # 新しい指令をすぐ方策に渡すため、観測の指令部分を書き換える
                obs["policy"][:, 6:9] = command * env.commands_scale

                if command.tolist() != last_command:
                    last_command = command.tolist()
                    print(f"\rcmd  vx={last_command[0]:+.2f} m/s  vy={last_command[1]:+.2f} m/s  "
                          f"wz={last_command[2]:+.2f} rad/s   ", end="", flush=True)

                obs, _, dones, _ = env.step(policy(obs))
                if switcher is not None:
                    switcher.after_step(dones)
                if dones.any():
                    # 転倒・地形の端でのリセット後はカメラも新しい位置へすぐ移す
                    apply_camera_mode()
                if state["camera_mode"] == "chase":
                    update_chase_camera()
                else:
                    # 位置だけ追従 / 固定: マウス操作の中心をロボットに置く (追従ならカメラも平行移動)
                    orbit.update(env.base_pos[0].cpu().numpy(), follow=state["camera_mode"] == "follow")
        except KeyboardInterrupt:
            pass
    print()


if __name__ == "__main__":
    main()
