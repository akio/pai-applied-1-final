"""宙返り (train.py --flip) の動画を 4 方向の視点で撮る。

1 回宙返りさせて関節角と胴体の位置・姿勢を記録し、それを再生しながら 4 台のカメラで撮った画面を 2 x 2 に並べる。
  左上: 真横 (回転の面を正面から)   右上: 正面
  左下: 斜め後ろの上から             右下: 追従するアップ
前半は等速、後半は踏み切りから着地までをスロー再生する。スローの中間のコマは、記録した 2 コマの間の姿勢を補間して作る
(制御周期 50 Hz のままでも滑らかに見えるように)。

    python go2_rough/record_flip.py -e go2-sprawling-strong-flip-v3 --ckpt 2600      # -> videos/go2-sprawling-strong_backflip_ckpt<N>.mp4
"""

import argparse
import math
import os
import pickle
import subprocess
import sys

import cv2
import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import genesis as gs

from env import robot_name
from flip import Go2FlipEnv
from policy_switch import load_policy

# 宙返りの指令を出す時刻 (エピソード開始から) [s]
TRIGGER_TIME = 0.6


def slerp(q0, q1, alpha):
    """クォータニオン (w, x, y, z) の球面線形補間。"""
    dot = torch.dot(q0, q1)
    if dot < 0.0:
        q1, dot = -q1, -dot
    if dot > 0.9995:
        q = q0 + alpha * (q1 - q0)
        return q / torch.norm(q)
    theta = torch.acos(dot)
    return (torch.sin((1.0 - alpha) * theta) * q0 + torch.sin(alpha * theta) * q1) / torch.sin(theta)


def interpolate_qpos(qpos, t, dt):
    """記録した qpos の列 (dt 間隔) から時刻 t の qpos を作る (位置と関節角は線形、胴体の姿勢は球面線形補間)。"""
    i = min(int(t / dt), len(qpos) - 2)
    alpha = min(max(t / dt - i, 0.0), 1.0)
    q0, q1 = qpos[i], qpos[i + 1]
    q = q0 + alpha * (q1 - q0)
    q[3:7] = slerp(q0[3:7], q1[3:7], alpha)
    return q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-sprawling-strong-flip-v3")
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--res", type=int, nargs=2, default=[960, 540], metavar=("W", "H"), help="1 視点の画面の大きさ")
    parser.add_argument("--slow", type=float, default=0.25, help="スロー再生の速さ (1 が等速)")
    parser.add_argument("--checker-size", type=float, default=0.5, help="地面の市松模様の 1 マスの大きさ [m]")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, _ = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    terrain_cfg["checker_size"] = args.checker_size

    gs.init(backend=gs.gpu, logging_level="warning")
    env = Go2FlipEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, camera_res=tuple(args.res))
    policy = load_policy(env, args.exp_name, args.ckpt)
    flip_cfg = command_cfg["flip"]

    # --- 1 回宙返りさせて記録する
    obs = env.reset()
    env.flip_trigger_time[:] = TRIGGER_TIME
    qpos = [env.robot.get_qpos()[0].clone()]
    with torch.no_grad():
        for _ in range(env.max_episode_length - 2):
            obs, _, _, _ = env.step(policy(obs))
            qpos.append(env.robot.get_qpos()[0].clone())
    rotation = env.flip_rotation[0].item() / (2.0 * math.pi)
    print(f"rotation: {rotation:.2f} turns")

    # --- カメラの配置: 指令を出したときの胴体の位置と向きを基準にする (回転の面は前後方向と上下方向)
    i_trigger = int(TRIGGER_TIME / env.dt)
    q_trigger = qpos[i_trigger].cpu().numpy()
    w, x, y, z = q_trigger[3:7]
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    forward = np.array([math.cos(yaw), math.sin(yaw), 0.0])
    left = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
    up = np.array([0.0, 0.0, 1.0])
    origin = np.array([q_trigger[0], q_trigger[1], env.env_origins[0, 2].item()])
    # 宙返りの最高点 (胴体が地面から 0.7 m 前後) まで画面に収まる高さを向く
    center = origin + 0.5 * up
    fixed_views = [
        ("side", center - 2.4 * left - 0.1 * up, center),
        ("front", center + 2.4 * forward, center),
        ("rear, from above", center - 1.5 * forward - 1.0 * left + 1.2 * up, center),
    ]

    width, height = args.res
    output = args.output or os.path.join(ROOT, "videos", f"{robot_name(env_cfg)}_backflip_ckpt{args.ckpt}.mp4")
    os.makedirs(os.path.dirname(output), exist_ok=True)
    fps = round(1.0 / env.dt)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{2 * width}x{2 * height}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output],
        stdin=subprocess.PIPE,
    )

    follow = {"lookat": None}

    def render_frame(t, speed_label):
        q = interpolate_qpos(qpos, t, env.dt)
        env.robot.set_qpos(q[None], zero_velocity=True)
        # step を進めずに変えた姿勢は、描画の状態を明示的に更新しないとカメラに映らない
        env.scene.visualizer.update(force=True)
        base = q[:3].cpu().numpy()
        follow["lookat"] = base if follow["lookat"] is None else 0.7 * follow["lookat"] + 0.3 * base
        views = fixed_views + [
            ("follow (close-up)", follow["lookat"] + 1.1 * forward - 1.0 * left + 0.25 * up, follow["lookat"])
        ]
        tiles = []
        for name, pos, lookat in views:
            env.camera.set_pose(pos=pos, lookat=lookat, up=(0.0, 0.0, 1.0))
            rgb = np.ascontiguousarray(env.camera.render()[0])
            rgb[:44, :] = (rgb[:44, :] * 0.45).astype(np.uint8)
            cv2.putText(rgb, name, (14, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
            tiles.append(rgb)
        frame = np.concatenate(
            (np.concatenate(tiles[:2], axis=1), np.concatenate(tiles[2:], axis=1)), axis=0
        )
        flip_t = t - TRIGGER_TIME
        phase = "prepare"
        for name, start in (
            ("push off + rotate", flip_cfg["prepare_s"]),
            ("flight", flip_cfg["prepare_s"] + flip_cfg["push_s"]),
            ("land", flip_cfg["land_s"] - 0.3),
            ("stand", flip_cfg["land_s"]),
        ):
            if flip_t >= start:
                phase = name
        if flip_t < 0.0:
            phase = "stand"
        label = f"{speed_label}   t = {flip_t:+.2f} s from flip command   [{phase}]"
        (text_w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
        # 画面下の中央に、半透明の黒帯の上に白文字で書く
        x0, y0 = width - text_w // 2, 2 * height - 22
        box = frame[y0 - 32 : y0 + 14, x0 - 14 : x0 + text_w + 14]
        box[:] = (box * 0.3).astype(np.uint8)
        cv2.putText(frame, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        ffmpeg.stdin.write(frame.tobytes())

    duration = (len(qpos) - 1) * env.dt
    # 等速: 指令の少し前から立ち直るまで
    for t in np.arange(TRIGGER_TIME - 0.3, duration, env.dt):
        render_frame(t, "x1.0")
    # スロー: 踏み切りの少し前から着地まで
    follow["lookat"] = None
    slow_start = TRIGGER_TIME + flip_cfg["prepare_s"] - 0.2
    slow_end = TRIGGER_TIME + flip_cfg["land_s"] + 0.2
    for t in np.arange(slow_start, slow_end, env.dt * args.slow):
        render_frame(t, f"x{args.slow:g} slow")

    ffmpeg.stdin.close()
    ffmpeg.wait()
    print(f"saved {output}")


if __name__ == "__main__":
    main()
