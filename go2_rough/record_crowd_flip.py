"""たくさんのロボットが一斉に後方宙返りする様子で、宙返りの学習の進み具合を見せる動画を作る。

平地に同じ向きのロボットを格子状に並べ、宙返りの指令を前から後ろへ波のようにずらして出す
(指令の時刻は学習のときと同じ 0.5〜1.0 s の範囲)。チェックポイントごとに 1 エピソードずつ撮って並べ、
最後に最終の方策を、回り込むカメラの等速と、真横に近い低い視点のスローで見せる。
宙返りの学習は 3 段階に分けて続けた (STAGES)。チェックポイントの番号は通しの iteration で、段階は自動で選ぶ。

1 エピソードをシミュレーションして全台の姿勢を記録し、それを再生しながら撮る (スローの中間のコマは補間する)。

    python go2_rough/record_crowd_flip.py                     # -> videos/go2-sprawling-strong_backflip_crowd_ckpt<N>.mp4
    python go2_rough/record_crowd_flip.py --num-envs 256 --ckpts 0 1300 2999
"""

import argparse
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs

from env import robot_name
from flip import Go2FlipEnv
from policy_switch import load_policy
from record import draw_progress_bar, put_text_band, title_frames

# 宙返りの学習の段階: (実験名, 最初の iteration, 最後の iteration, 前の段階からの変更)
STAGES = [
    ("go2-sprawling-strong-flip", 0, 1300, "stage 1"),
    ("go2-sprawling-strong-flip-v2", 1300, 2000, "stage 2: higher cap on the rotation-speed reward"),
    ("go2-sprawling-strong-flip-v3", 2000, 2999, "stage 3: large penalty for the body touching the ground"),
]
DEFAULT_CKPTS = [0, 100, 300, 600, 1000, 1300, 1600, 2000, 2300, 2600, 2999]
TITLE_SECONDS = 2.5
HOLD_SECONDS = 1.0  # クリップの最後のコマを止めて結果を見せる秒数
SPACING = 1.3  # 格子の間隔 [m]


def stage_of(ckpt):
    for i, (exp_name, first, last, _) in enumerate(STAGES):
        if first <= ckpt <= last and (i == 0 or ckpt > first):
            return i
    raise ValueError(f"iteration {ckpt} は STAGES のどの段階にもない")


class GridFlipEnv(Go2FlipEnv):
    """リセットのたびに、ロボットを決まった格子の位置に +x 向きで置き、宙返りの指令を前の列から順に出す Go2FlipEnv。"""

    # 格子の位置 (N, 2) と、指令を出す時刻の順番 (N,) 0〜1。作ったあとに外から与える (それまでは普通にリセットする)
    grid_xy = None
    grid_wave = None

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        super()._reset_idx(env_ids)
        if self.grid_xy is None:
            return
        n = len(env_ids)
        xy = self.grid_xy[env_ids]
        ground = self._terrain_heights(xy)
        pos = torch.cat((xy, (ground + self.env_cfg["base_init_pos"][2])[:, None]), dim=-1)
        quat = self.init_base_quat.expand(n, -1)
        qpos = torch.cat((pos, quat, self.init_dof_pos_qpos_order.expand(n, -1)), dim=-1)
        self.robot.set_qpos(qpos, envs_idx=env_ids, zero_velocity=True)
        self.env_origins[env_ids, :2] = xy
        self.env_origins[env_ids, 2] = ground
        self.base_pos[env_ids] = pos
        self.base_quat[env_ids] = quat
        self.base_euler[env_ids] = 0.0
        self.feet_pos[env_ids] = self._get_feet_pos()[env_ids]
        self.com_pos[env_ids] = self._get_com_pos()[env_ids]
        low, high = self.flip_cfg["trigger_range_s"]
        self.flip_trigger_time[env_ids] = low + (high - low) * self.grid_wave[env_ids]


def simulate(env, policy):
    """1 エピソード分の全台の qpos (T, N, 19)、回転数の平均の推移 (T,)、成功 (N,) を返す。成功の条件は eval.py と同じ。"""
    obs = env.reset()
    n = env.num_envs
    window = env.jump_cfg["window_s"]
    qpos = [env.robot.get_qpos().clone()]
    mean_turns = [0.0]
    has_base_contact = torch.zeros(n, dtype=torch.bool, device=gs.device)
    is_upright = torch.zeros(n, dtype=torch.bool, device=gs.device)
    rotation = torch.zeros(n, device=gs.device)
    with torch.no_grad():
        for _ in range(env.max_episode_length - 2):
            obs, _, _, _ = env.step(policy(obs))
            t = env.flip_time
            is_active = (t >= 0.0) & (t < window)
            has_base_contact |= is_active & (torch.norm(env.contact_forces[:, env.base_link_idx], dim=-1) > 1.0)
            is_end = (t >= window) & (t < window + env.dt)
            is_upright = torch.where(is_end, env.projected_gravity[:, 2] < -math.cos(math.radians(25)), is_upright)
            rotation = torch.where(is_end, env.flip_rotation / (2.0 * math.pi), rotation)
            qpos.append(env.robot.get_qpos().clone())
            mean_turns.append((env.flip_rotation / (2.0 * math.pi)).mean().item())
    is_success = (rotation > 0.75) & (rotation < 1.25) & is_upright & ~has_base_contact
    return torch.stack(qpos), mean_turns, is_success


# カメラ: (格子の中心, 格子の x 方向の半幅, y 方向の半幅, 時刻 0〜1) -> (位置, 注視点)
def side_camera(center, half_w, half_d, a):
    """手前 (-y 側) の斜め上から、手前の数列を大きく、奥へ並ぶ列も見える向き。クリップの中で少しずつ横に動く。"""
    lookat = center + np.array([(a - 0.5) * 0.3 * half_w, -half_d + 3.0, 0.2])
    return lookat + np.array([-1.5, -6.5, 3.2]), lookat


def orbit_camera(center, half_w, half_d, a):
    """格子の全体が入る距離で、手前から斜めへ回り込む。"""
    angle = math.radians(-90.0 - 50.0 * a)
    radius = 0.75 * max(half_w, half_d) + 4.0
    lookat = center + np.array([0.0, -0.3 * half_d, 0.3])
    return lookat + np.array([radius * math.cos(angle), radius * math.sin(angle), 0.35 * radius]), lookat


def low_camera(center, half_w, half_d, a):
    """真横の低い位置から、手前の数列をアップで。"""
    lookat = center + np.array([(a - 0.5) * 0.3 * half_w, -half_d + 1.5, 0.32])
    return lookat + np.array([0.0, -5.0, 0.9]), lookat


def interpolate_qpos(qpos, t, dt):
    """記録した qpos (T, N, 19) から時刻 t の姿勢を作る (位置・関節角は線形、姿勢のクォータニオンは正規化した線形補間)。"""
    i = min(int(t / dt), len(qpos) - 2)
    alpha = min(max(t / dt - i, 0.0), 1.0)
    q0, q1 = qpos[i], qpos[i + 1]
    quat1 = torch.where((q0[:, 3:7] * q1[:, 3:7]).sum(dim=1, keepdim=True) < 0.0, -q1[:, 3:7], q1[:, 3:7])
    q = q0 + alpha * (q1 - q0)
    quat = q0[:, 3:7] + alpha * (quat1 - q0[:, 3:7])
    q[:, 3:7] = quat / torch.norm(quat, dim=1, keepdim=True)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts", type=int, nargs="+", default=DEFAULT_CKPTS, help="通しの iteration")
    parser.add_argument("--num-envs", type=int, default=400, help="ロボットの台数 (格子に並べて全員を描画する)")
    parser.add_argument("--slow", type=float, default=0.25, help="最後のスローの速さ (1 が等速)")
    parser.add_argument("--checker-size", type=float, default=1.0, help="地面の市松模様の 1 マスの大きさ [m]")
    parser.add_argument("--res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    # 環境は最後の段階の設定で作る (観測・ロボットはどの段階も同じ。報酬は使わない)
    with open(os.path.join(ROOT, "logs", STAGES[-1][0], "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, _ = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    terrain_cfg["checker_size"] = args.checker_size
    ckpts = sorted(args.ckpts)
    last_ckpt = STAGES[-1][2]
    output = args.output or os.path.join(
        ROOT, "videos", f"{robot_name(env_cfg)}_backflip_crowd_ckpt{ckpts[-1]}.mp4"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")
    n = args.num_envs
    env = GridFlipEnv(
        n, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg,
        camera_res=tuple(args.res), rendered_envs_idx=range(n), camera_far=200.0,
    )
    # 格子: x (ロボットの前後) に cols 列、y に rows 行。地形の中心に置く
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    ids = torch.arange(n, device=gs.device)
    ix, iy = ids % cols, ids // cols
    center = env.terrain_extent / 2
    env.grid_xy = torch.stack(
        (center[0] + (ix - (cols - 1) / 2) * SPACING, center[1] + (iy - (rows - 1) / 2) * SPACING), dim=-1
    )
    # 前の列 (x が大きい) から順に指令を出す
    env.grid_wave = 1.0 - ix.float() / max(cols - 1, 1)
    center = np.array([center[0].item(), center[1].item(), 0.0])
    half_w, half_d = (cols - 1) / 2 * SPACING, (rows - 1) / 2 * SPACING

    width, height = args.res
    fps = round(1.0 / env.dt)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output],
        stdin=subprocess.PIPE,
    )

    def write_title(lines):
        for rgb in title_frames(width, height, lines, TITLE_SECONDS, fps):
            ffmpeg.stdin.write(rgb.tobytes())

    def render(q, pos, lookat):
        env.robot.set_qpos(q, zero_velocity=True)
        # step を進めずに変えた姿勢は、描画の状態を明示的に更新しないとカメラに映らない
        env.scene.visualizer.update(force=True)
        env.camera.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))
        return np.ascontiguousarray(env.camera.render()[0])

    def stage_label(ckpt):
        return f"{STAGES[stage_of(ckpt)][3]}"

    def put_labels(rgb, ckpt, extra, result=None):
        line2 = extra if result is None else f"{extra}   successful flips: {result}"
        put_text_band(rgb, (f"iteration {ckpt} | {stage_label(ckpt)}", line2))
        draw_progress_bar(rgb, ckpt, last_ckpt)

    def run_clip(ckpt, times, camera, speed_label=""):
        """times: 再生する時刻 [s] の列 (エピソード開始から)。最後のコマは HOLD_SECONDS だけ止めて結果を出す。"""
        exp_name = STAGES[stage_of(ckpt)][0]
        policy = load_policy(env, exp_name, ckpt)
        qpos, mean_turns, is_success = simulate(env, policy)
        n_success = int(is_success.sum().item())
        result = f"{n_success} / {n} ({100 * n_success / n:.0f}%)"
        rgb = None
        for k, t in enumerate(times):
            pos, lookat = camera(center, half_w, half_d, k / max(len(times) - 1, 1))
            rgb = render(interpolate_qpos(qpos, t, env.dt), pos, lookat)
            turns = mean_turns[min(int(round(t / env.dt)), len(mean_turns) - 1)]
            put_labels(rgb, ckpt, f"{speed_label}mean turns: {turns:.2f}")
            ffmpeg.stdin.write(rgb.tobytes())
        put_labels(rgb, ckpt, f"{speed_label}mean turns: {mean_turns[-1]:.2f}", result)
        for _ in range(int(HOLD_SECONDS * fps)):
            ffmpeg.stdin.write(rgb.tobytes())
        print(f"iteration {ckpt} ({exp_name}) | mean turns {mean_turns[-1]:.2f} | success {result}")

    episode_s = (env.max_episode_length - 2) * env.dt
    realtime = np.arange(0.3, episode_s, env.dt)

    write_title(
        (
            ("Go2 Sprawling (2x torque): learning a back flip", 1.0),
            (f"{n} robots, flip command sent row by row (0.5-1.0 s after start)", 0.7),
            ("success: 0.75-1.25 turns, upright 2 s after the command, body never touches the ground", 0.6),
        )
    )
    for ckpt in ckpts:
        run_clip(ckpt, realtime, side_camera)
    write_title(((f"final policy (iteration {ckpts[-1]})", 1.0),))
    run_clip(ckpts[-1], realtime, orbit_camera)
    # スロー: 最初の列の踏み切りの少し前から、最後の列の着地まで
    flip_cfg = command_cfg["flip"]
    low, high = flip_cfg["trigger_range_s"]
    slow_times = np.arange(low + flip_cfg["prepare_s"] - 0.2, high + flip_cfg["land_s"], env.dt * args.slow)
    run_clip(ckpts[-1], slow_times, low_camera, f"x{args.slow:g} slow   ")

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
