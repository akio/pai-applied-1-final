"""たくさんのロボットを 1 つの画面に写して、学習の進み具合を見せる動画を作る。

不整地の全マス (地形 6 種類 x 難易度 8 段階) にロボットを均等に置き、学習のときと同じように
ランダムな速度指令で歩かせる (転んだロボットは自分のマスの中心に戻る)。
チェックポイントごとに同じカメラの動き (易しい難易度から難しい難易度へ横に移動) で撮って並べ、
最後に最終の方策で、近くから地形全体が見える高さまでカメラを引いていく場面を入れる。

    python go2_rough/record_crowd.py                           # -> videos/go2-sprawling_rough-terrain_crowd_ckpt<N>.mp4
    python go2_rough/record_crowd.py --num-envs 384 --ckpts 0 500 2999
"""

import argparse
import glob
import math
import os
import pickle
import re
import subprocess
import sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import TERRAIN_TYPES, Go2RoughEnv, robot_name
from record import draw_progress_bar, put_text_band, title_frames

TITLE_SECONDS = 2.5
# 動画のコマは制御 2 ステップに 1 枚 (25 fps)
STEPS_PER_FRAME = 2


def smoothstep(a):
    a = min(max(a, 0.0), 1.0)
    return a * a * (3.0 - 2.0 * a)


def progress_camera(env, a):
    """チェックポイントごとのクリップのカメラ (a: クリップの中の時刻 0〜1)。難易度の低い側から高い側へ横に動く。"""
    size, border = env.terrain_cfg["subterrain_size"], env.terrain_cfg["border_size"]
    n_types = len(TERRAIN_TYPES)
    x = border + size * (1.5 + 4.0 * smoothstep(a))  # 難易度 1 の中心 -> 難易度 5 の中心
    lookat = np.array([x, border + size * n_types * 0.45, 0.0])
    return lookat + np.array([-2.0, -10.0, 6.5]), lookat


def final_camera(env, a):
    """最後のクリップのカメラ。地形の中心の周りを回りながら、近くから地形全体が見える高さまで引く。"""
    extent = env.terrain_extent.cpu().numpy()
    center = np.array([extent[0] / 2, extent[1] / 2, 0.0])
    s = smoothstep(a)
    angle = math.radians(-120.0 + 70.0 * a)
    radius = 9.0 + 31.0 * s
    lookat = center + np.array([6.0 * (1.0 - s), 0.0, 0.0])
    pos = lookat + np.array([radius * math.cos(angle), radius * math.sin(angle), 4.0 + 26.0 * s])
    return pos, lookat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-sprawling-rough-fast")
    parser.add_argument("--ckpts", type=int, nargs="+", default=None, help="省略時: 最後の iteration の約 0/3/10/20/40/70/100%%")
    parser.add_argument("--num-envs", type=int, default=768, help="ロボットの台数 (全員を描画する)")
    parser.add_argument("--clip-seconds", type=float, default=6.0, help="チェックポイント 1 つあたりの秒数")
    parser.add_argument("--final-seconds", type=float, default=12.0, help="最後のクリップの秒数")
    parser.add_argument("--res", type=int, nargs=2, default=[1280, 720], metavar=("W", "H"))
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    # 難易度は固定して全マスにロボットがいるようにする。速度指令の範囲は学習の初めの範囲 (前後 ±1 m/s) のまま
    terrain_cfg["curriculum"] = False
    command_cfg["curriculum"] = False
    # 1 マスに十数台を置くので、出発位置をマスの中心から ±2 m に広げて重なりを減らす (学習のときは ±0.5 m)
    env_cfg["spawn_spread"] = 2.0

    saved_ckpts = sorted(int(re.search(r"model_(\d+)\.pt", p).group(1)) for p in glob.glob(f"{log_dir}/model_*.pt"))
    last_ckpt = saved_ckpts[-1]
    ratios = (0.0, 0.03, 0.1, 0.2, 0.4, 0.7, 1.0)
    ckpts = args.ckpts or sorted({min(saved_ckpts, key=lambda c: abs(c - last_ckpt * r)) for r in ratios})
    output = args.output or os.path.join(
        ROOT, "videos", f"{robot_name(env_cfg)}_rough-terrain_crowd_ckpt{ckpts[-1]}.mp4"
    )
    os.makedirs(os.path.dirname(output), exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")
    n = args.num_envs
    env = Go2RoughEnv(
        num_envs=n,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        camera_res=tuple(args.res),
        rendered_envs_idx=range(n),
        camera_far=200.0,
    )
    ids = torch.arange(n, device=gs.device)
    n_types = len(TERRAIN_TYPES)
    env.terrain_types[:] = ids % n_types
    env.terrain_levels[:] = (ids // n_types) % terrain_cfg["n_levels"]
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    width, height = args.res
    fps = round(1.0 / (env.dt * STEPS_PER_FRAME))
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output],
        stdin=subprocess.PIPE,
    )

    def write_title(lines):
        for rgb in title_frames(width, height, lines, TITLE_SECONDS, fps):
            ffmpeg.stdin.write(rgb.tobytes())

    def run_clip(ckpt, seconds, camera, label):
        runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
        policy = runner.get_inference_policy(device=gs.device)
        obs = env.reset()
        has_fallen = torch.zeros(n, dtype=torch.bool, device=gs.device)
        n_frames = int(seconds / (env.dt * STEPS_PER_FRAME))
        with torch.no_grad():
            for i_frame in range(n_frames):
                for _ in range(STEPS_PER_FRAME):
                    obs, _, dones, extras = env.step(policy(obs))
                    has_fallen |= dones.bool() & (extras["time_outs"] < 0.5)
                pos, lookat = camera(env, i_frame / max(n_frames - 1, 1))
                env.camera.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))
                rgb = np.ascontiguousarray(env.camera.render()[0])
                n_fallen = int(has_fallen.sum().item())
                # 出発位置 (転んだロボットは戻った位置) からの距離の平均
                moved = torch.norm(env.base_pos[:, :2] - env.env_origins[:, :2], dim=1).mean().item()
                put_text_band(
                    rgb, (label, f"average distance from start: {moved:.1f} m   fallen: {n_fallen} / {n} ({100 * n_fallen / n:.0f}%)")
                )
                draw_progress_bar(rgb, ckpt, last_ckpt)
                ffmpeg.stdin.write(rgb.tobytes())
        print(f"{label} | distance: {moved:.1f} m | fallen: {n_fallen} / {n}")

    write_title(
        (
            ("Go2 Sprawling: learning to walk on rough terrain", 1.0),
            (f"{n} robots at once, random velocity commands as in training", 0.7),
            ("left: easy levels -> right: hard levels", 0.7),
        )
    )
    for ckpt in ckpts:
        run_clip(ckpt, args.clip_seconds, progress_camera, f"iteration {ckpt} | {n} robots on 6 terrain types x 8 levels")
    write_title(((f"final policy (iteration {ckpts[-1]})", 1.0), ("the whole terrain", 0.7)))
    run_clip(ckpts[-1], args.final_seconds, final_camera, f"iteration {ckpts[-1]} | {n} robots on 6 terrain types x 8 levels")

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
