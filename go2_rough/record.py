"""学習済み方策の歩行を mp4 に録画する。

progress モード (デフォルト): 同じ地形で複数のチェックポイントを順に撮り、学習の進み具合を比べる動画。
tour モード (--tour): 最後のチェックポイントで各地形タイプを順に撮る動画。
story モード (--story): 学習の過程を短く見せたあと、最後のチェックポイントで地形・速度を変えた場面を長めに見せる動画。
1 クリップごとに左上へ iteration / 地形 / 難易度を書き込む。story モードでは画面下に学習の進み具合のバーも出す。
"""

import argparse
import glob
import math
import os
import pickle
import re
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import TERRAIN_TYPES, robot_name
from eval import FixedCommandEnv

# tour モードで撮る (地形, 難易度, 指令)
TOUR = [
    ("stairs_up", 6, (0.6, 0.0, 0.0)),
    ("stairs_down", 6, (0.6, 0.0, 0.0)),
    ("slope_up", 6, (0.6, 0.0, 0.0)),
    ("obstacles", 6, (0.6, 0.0, 0.0)),
    ("rough", 7, (0.0, 0.4, 0.0)),
    ("rough", 7, (0.0, 0.0, 1.0)),
]
# story モード: 学習の過程を見せる地形と、1 チェックポイントあたりの秒数
STORY_PROGRESS = ("stairs_up", 6, (0.6, 0.0, 0.0), 3.0)
# story モード: 最後のチェックポイントで撮る (地形, 難易度, 指令, 秒数, 向き)。向きはワールド x 軸からの角度で、
# pi は難易度の低い側 (同じ地形タイプの中) へ進む向き。0 は難易度の高い側へ進む向き
STORY_FINAL = [
    ("stairs_up", 7, (0.6, 0.0, 0.0), 7.0, math.pi),
    ("stairs_down", 7, (0.6, 0.0, 0.0), 7.0, math.pi),
    ("slope_up", 7, (0.8, 0.0, 0.0), 6.0, math.pi),
    ("slope_down", 7, (0.8, 0.0, 0.0), 6.0, math.pi),
    ("obstacles", 7, (0.6, 0.0, 0.0), 7.0, math.pi),
    ("rough", 7, (0.8, 0.0, 0.0), 6.0, math.pi),
    ("rough", 7, (0.0, 0.5, 0.0), 6.0, math.pi / 2),
    ("rough", 5, (0.5, 0.0, 0.8), 7.0, math.pi),
    ("stairs_up", 4, (-0.6, 0.0, 0.0), 6.0, 0.0),
    ("slope_up", 0, (2.0, 0.0, 0.0), 4.0, 0.0),
]
TITLE_SECONDS = 2.5


def put_text_band(rgb, lines, top=0, height=90):
    """半透明の黒帯の上に白文字で複数行を書く (明るい地形の上でも読めるように)。"""
    rgb[top : top + height, :] = (rgb[top : top + height, :] * 0.4).astype(np.uint8)
    for i, text in enumerate(lines):
        cv2.putText(rgb, text, (20, top + 38 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_progress_bar(rgb, iteration, last_iteration):
    """画面下に学習の進み具合 (iteration / 最後の iteration) のバーを書く。"""
    h, w = rgb.shape[:2]
    x0, x1, y0, y1 = 20, w - 20, h - 34, h - 18
    rgb[y0 - 30 : h, :] = (rgb[y0 - 30 : h, :] * 0.4).astype(np.uint8)
    cv2.rectangle(rgb, (x0, y0), (x1, y1), (120, 120, 120), -1)
    cv2.rectangle(rgb, (x0, y0), (x0 + int((x1 - x0) * iteration / last_iteration), y1), (80, 200, 255), -1)
    text = f"training progress: iteration {iteration} / {last_iteration}"
    cv2.putText(rgb, text, (x0, y0 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)


def title_frames(width, height, lines, seconds, fps):
    """黒地に白文字のタイトル画面のコマを返す。"""
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    for i, (text, scale) in enumerate(lines):
        (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
        y = height // 2 - 30 * (len(lines) - 1) + 60 * i + text_h // 2
        cv2.putText(rgb, text, ((width - text_w) // 2, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 2, cv2.LINE_AA)
    return [rgb] * int(seconds * fps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-rough")
    parser.add_argument("--ckpts", type=int, nargs="+", default=None, help="省略時: progress は 0/200/500/1000/最後、story は約 10 段階")
    parser.add_argument("--terrain", choices=TERRAIN_TYPES, default="stairs_up")
    parser.add_argument("--level", type=int, default=5)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.6, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--tour", action="store_true")
    parser.add_argument("--story", action="store_true")
    parser.add_argument("--clip-seconds", type=float, default=6.0, help="progress / tour モードの 1 クリップの秒数")
    parser.add_argument("--res", type=int, nargs=2, default=[960, 540], metavar=("W", "H"))
    parser.add_argument("-o", "--output", type=str, default=None)
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    terrain_cfg["curriculum"] = False
    command_cfg["curriculum"] = False
    # クリップの途中でエピソードが打ち切られないようにする
    env_cfg["episode_length_s"] = 1e6

    saved_ckpts = sorted(int(re.search(r"model_(\d+)\.pt", p).group(1)) for p in glob.glob(f"{log_dir}/model_*.pt"))
    last_ckpt = saved_ckpts[-1]
    # クリップ: (チェックポイント, 地形, 難易度, 指令, 秒数, 向き (None ならランダム))
    # titles: クリップの番号 -> そのクリップの前に入れるタイトル画面の行 ((文字列, 文字の大きさ), ...)
    titles = {}
    if args.story:
        # 学習の序盤ほど変化が大きいので、序盤を細かく選ぶ
        ratios = (0.0, 0.03, 0.07, 0.1, 0.17, 0.27, 0.4, 0.55, 0.7, 0.85)
        ckpts = args.ckpts or sorted({min(saved_ckpts, key=lambda c: abs(c - last_ckpt * r)) for r in ratios})
        terrain, level, cmd, seconds = STORY_PROGRESS
        clips = [(ckpt, terrain, level, cmd, seconds, math.pi) for ckpt in ckpts]
        titles[0] = (
            ("Go2 Sprawling: learning to walk on rough terrain", 1.0),
            (f"training progress on {terrain} lv{level}, {seconds:.0f} s per checkpoint", 0.7),
        )
        titles[len(clips)] = (
            (f"final policy (iteration {last_ckpt})", 1.0),
            ("hardest levels, sideways / turning / backward, 2 m/s", 0.7),
        )
        clips += [(last_ckpt, t, lv, c, sec, yaw) for t, lv, c, sec, yaw in STORY_FINAL]
        default_name = f"{robot_name(env_cfg)}_rough-terrain_story_ckpt{last_ckpt}.mp4"
    elif args.tour:
        ckpt = (args.ckpts or [last_ckpt])[-1]
        clips = [(ckpt, terrain, level, cmd, args.clip_seconds, None) for terrain, level, cmd in TOUR]
        default_name = f"{robot_name(env_cfg)}_rough-terrain_tour_ckpt{ckpt}.mp4"
    else:
        ckpts = args.ckpts or [0, 200, 500, 1000, last_ckpt]
        clips = [(ckpt, args.terrain, args.level, tuple(args.cmd), args.clip_seconds, None) for ckpt in ckpts]
        default_name = f"{robot_name(env_cfg)}_progress_{args.terrain}_lv{args.level}.mp4"
    output = args.output or os.path.join(ROOT, "videos", default_name)
    os.makedirs(os.path.dirname(output), exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")
    env = FixedCommandEnv(
        num_envs=1,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        camera_res=tuple(args.res),
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)

    width, height = args.res
    fps = round(1.0 / env.dt)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "20", output],
        stdin=subprocess.PIPE,
    )

    for i_clip, (ckpt, terrain, level, cmd, seconds, yaw) in enumerate(clips):
        if i_clip in titles:
            for rgb in title_frames(width, height, titles[i_clip], TITLE_SECONDS, fps):
                ffmpeg.stdin.write(rgb.tobytes())
        runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
        policy = runner.get_inference_policy(device=gs.device)
        FixedCommandEnv.fixed_command = cmd
        env.terrain_types[:] = TERRAIN_TYPES.index(terrain)
        env.terrain_levels[:] = level
        obs = env.reset()
        if yaw is not None:
            # 地形の中で進む向きをそろえる (ランダムな向きだと隣の別の地形タイプへ出てしまう)
            quat = torch.tensor([[math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)]], device=gs.device)
            env.robot.set_quat(quat, zero_velocity=True)
        else:
            yaw = math.radians(env.base_euler[0, 2].item())

        # 出発時の向きに対して斜め後ろからの視点に固定し、位置だけ追従する
        rot = np.array([[math.cos(yaw), -math.sin(yaw)], [math.sin(yaw), math.cos(yaw)]])
        offset = np.append(rot @ np.array([-1.3, -1.1]), 0.8)
        lookat = env.base_pos[0].cpu().numpy()
        start_xy = lookat[:2].copy()
        n_falls = 0
        label = f"iter {ckpt} | {terrain} lv{level} | cmd vx={cmd[0]:.1f} vy={cmd[1]:.1f} wz={cmd[2]:.1f}"
        with torch.no_grad():
            for _ in range(int(seconds / env.dt)):
                obs, _, dones, extras = env.step(policy(obs))
                n_falls += int((dones & (extras["time_outs"] < 0.5)).item())
                lookat = 0.9 * lookat + 0.1 * env.base_pos[0].cpu().numpy()
                env.camera.set_pose(pos=lookat + offset, lookat=lookat, up=(0.0, 0.0, 1.0))
                rgb = np.ascontiguousarray(env.camera.render()[0])
                moved = np.linalg.norm(env.base_pos[0, :2].cpu().numpy() - start_xy)
                put_text_band(rgb, (label, f"falls: {n_falls}   distance from start: {moved:.1f} m"))
                if args.story:
                    draw_progress_bar(rgb, ckpt, last_ckpt)
                ffmpeg.stdin.write(rgb.tobytes())
        print(f"{label} | falls: {n_falls} | distance: {moved:.1f} m")

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved: {output}")


if __name__ == "__main__":
    main()
