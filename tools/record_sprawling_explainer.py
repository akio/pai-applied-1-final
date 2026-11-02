"""元の Go2 から Go2 Sprawling への改造を解説する動画を作る。

画面の左に元の Go2、右に Go2 Sprawling を同じ視点で並べ、改造の 3 点を順に見せる。
  1. 付け根 (hip) の関節の軸を前後方向から鉛直に変えた
  2. 付け根を左右に 10 cm 外へ出し、脚を真横へ向けた (太もも (thigh) の軸が前後方向になる)
  3. 後ろ脚を前脚の前後反転にした (前後・左右とも対称)
最後に、それぞれの学習済み方策で平地を歩かせた様子を並べる。
関節の動きは物理計算をせずに関節角を直接与えて見せる (歩行の場面だけ、学習環境で歩かせた記録を再生する)。
寸法 (付け根の位置・足先の幅・胴体の高さ) は URDF から計算して表示する。

    python tools/record_sprawling_explainer.py      # -> videos/go2-sprawling_explainer.mp4
"""

import argparse
import math
import os
import pickle
import subprocess
import sys

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "go2_rough"))

import genesis as gs
from genesis.utils.geom import transform_by_quat

from eval import FixedCommandEnv
from policy_switch import load_policy

W, H = 1280, 720
HEADER_H = 110
PANEL_W, PANEL_H = 640, H - HEADER_H
FPS = 30
FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
FONT_REGULAR = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
BG = (18, 22, 30)
# 足先 (すねの先の球) の中心のすねのリンク上の位置と半径
FOOT_OFFSET = (0.0, 0.0, -0.213)
FOOT_RADIUS = 0.022
LEGS = ("FR", "FL", "RR", "RL")

# (表示名, URDF, 実験名, チェックポイント, 並べる位置の y)
ROBOTS = (
    ("元の Go2", "go2_orig.urdf", "go2-rough", 1999, -5.0),
    ("Go2 Sprawling", "go2.urdf", "go2-sprawling-rough-fast", 2999, 5.0),
)
WALKS = (
    ("前進 0.8 m/s", (0.8, 0.0, 0.0), 6.0),
    ("横歩き 0.5 m/s", (0.0, 0.5, 0.0), 5.0),
)


def font(size, bold=False):
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REGULAR, size, index=0)  # index 0 が日本語 (JP)


class Robot:
    """解説用シーンの中の 1 台。物理計算はせず、姿勢を直接与えて描画する。"""

    def __init__(self, scene, name, urdf, exp, ckpt, lane_y):
        self.name, self.exp, self.ckpt, self.lane_y = name, exp, ckpt, lane_y
        with open(os.path.join(ROOT, "logs", exp, "cfgs.pkl"), "rb") as f:
            self.cfgs = pickle.load(f)
        env_cfg = self.cfgs[0]
        self.joint_names = env_cfg["joint_names"]
        self.default = np.array([env_cfg["default_joint_angles"][n] for n in self.joint_names])
        self.entity = scene.add_entity(gs.morphs.URDF(file=os.path.join(ROOT, "go2_sprawling", "urdf", urdf)))

    def setup(self):
        self.dofs_idx = [self.entity.get_joint(n).dofs_idx_local[0] for n in self.joint_names]
        self.calf_links = [self.entity.get_link(f"{leg}_calf") for leg in LEGS]
        # 基準姿勢で足先が地面に触れる胴体の高さ
        self.pose(self.default, z=1.0)
        self.stand_height = 1.0 - min(p[2] for p in self.feet()) + FOOT_RADIUS
        self.pose(self.default)

    def pose(self, joints, x=0.0, z=None):
        """胴体を自分の列 (y = lane_y) の x に、向きを +x にして置き、関節角を与える。"""
        z = self.stand_height if z is None else z
        self.entity.set_pos(torch.tensor([x, self.lane_y, z], device=gs.device))
        self.entity.set_quat(torch.tensor([1.0, 0.0, 0.0, 0.0], device=gs.device))
        self.entity.set_dofs_position(torch.tensor(joints, dtype=gs.tc_float, device=gs.device), self.dofs_idx)

    def base(self):
        return self.entity.get_pos().cpu().numpy()

    def feet(self):
        """足先の位置 (FR, FL, RR, RL の順)。"""
        out = []
        for link in self.calf_links:
            quat = link.get_quat()
            offset = torch.tensor(FOOT_OFFSET, dtype=quat.dtype, device=quat.device)
            out.append((link.get_pos() + transform_by_quat(offset, quat)).cpu().numpy())
        return out

    def joint_axes(self, kind):
        """kind (hip / thigh / calf) の 4 関節の (位置, 軸の向き)。"""
        joints = [self.entity.get_joint(f"{leg}_{kind}_joint") for leg in LEGS]
        return [(j.get_anchor_pos().cpu().numpy(), j.get_anchor_axis().cpu().numpy()) for j in joints]


class Video:
    def __init__(self, scene, robots, output):
        self.scene, self.robots = scene, robots
        self.camera = scene.add_camera(res=(PANEL_W, PANEL_H), fov=40, debug=True)  # debug: 矢印などの目印も写す
        self.ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
             "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18", output],
            stdin=subprocess.PIPE,
        )
        self.n_frames = 0

    def render_panel(self, robot, cam_offset, lookat_offset, up=(0.0, 0.0, 1.0), center=None):
        center = robot.base() if center is None else center
        lookat = center + np.asarray(lookat_offset)
        self.camera.set_pose(pos=lookat + np.asarray(cam_offset), lookat=lookat, up=up)
        return np.ascontiguousarray(self.camera.render()[0])

    def frame(self, panels, title, desc, captions, fade=1.0):
        """panels: 左右の画面, captions: 左右の画面の下に書く説明 (行のリスト)。"""
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        draw.text((32, 14), title, font=font(34, bold=True), fill=(255, 255, 255))
        draw.text((34, 64), desc, font=font(22), fill=(200, 210, 225))
        for i, (panel, robot) in enumerate(zip(panels, self.robots)):
            x0 = i * PANEL_W
            img.paste(Image.fromarray(panel), (x0, HEADER_H))
            # 画面の左上に機体名、下に説明を半透明の帯で重ねる
            overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
            od = ImageDraw.Draw(overlay)
            od.rectangle((x0, HEADER_H, x0 + PANEL_W, HEADER_H + 44), fill=(0, 0, 0, 120))
            lines = captions[i] if captions else []
            if lines:
                band_h = 18 + 32 * len(lines)
                od.rectangle((x0, HEADER_H + PANEL_H - band_h, x0 + PANEL_W, HEADER_H + PANEL_H), fill=(0, 0, 0, 150))
            img.paste(overlay, (0, 0), overlay)
            draw.text((x0 + 16, HEADER_H + 6), robot.name, font=font(24, bold=True), fill=(255, 215, 90) if i else (150, 200, 255))
            for k, line in enumerate(lines):
                y = HEADER_H + PANEL_H - 18 - 32 * len(lines) + 32 * k + 6
                draw.text((x0 + 16, y), line, font=font(22), fill=(255, 255, 255))
        draw.line((PANEL_W, HEADER_H, PANEL_W, HEADER_H + PANEL_H), fill=BG, width=4)
        self.write(img, fade)

    def card(self, lines, seconds):
        """黒地に中央揃えの文字だけの画面。lines: ((文字列, 大きさ, 太字), ...)"""
        img = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(img)
        heights = [size * 1.6 for _, size, _ in lines]
        y = (H - sum(heights)) / 2
        for (text, size, bold), h in zip(lines, heights):
            f = font(size, bold)
            x0, _, x1, _ = draw.textbbox((0, 0), text, font=f)
            draw.text(((W - (x1 - x0)) / 2, y), text, font=f, fill=(255, 255, 255))
            y += h
        n = int(seconds * FPS)
        for k in range(n):
            self.write(img, min(1.0, k / 8, (n - k) / 8))

    def write(self, img, fade=1.0):
        rgb = np.asarray(img)
        if fade < 1.0:
            rgb = (rgb * max(fade, 0.0)).astype(np.uint8)
        self.ffmpeg.stdin.write(rgb.tobytes())
        self.n_frames += 1

    def close(self):
        self.ffmpeg.stdin.close()
        self.ffmpeg.wait()


def section_frames(seconds):
    """場面の各コマの (経過時間, フェードの係数)。"""
    n = int(seconds * FPS)
    return [(k / FPS, min(1.0, k / 8, (n - k) / 8)) for k in range(n)]


def draw_axes(scene, robots, kind, color, length=0.36):
    # 関節は部品の中にあるので、矢印を長くして部品の外まで出す
    for robot in robots:
        for pos, axis in robot.joint_axes(kind):
            scene.draw_debug_arrow(pos - 0.5 * length * axis, length * axis, radius=0.008, color=color)


def record_walk(robot_def, cmd, seconds):
    """学習環境 (平地) で方策を歩かせ、胴体の位置・姿勢と関節角 (qpos) を記録する。"""
    _, _, exp, ckpt, _ = robot_def
    with open(os.path.join(ROOT, "logs", exp, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, _ = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    command_cfg["curriculum"] = False
    terrain_cfg["curriculum"] = False
    terrain_cfg["flat"] = True
    env_cfg["episode_length_s"] = 1e6
    FixedCommandEnv.fixed_command = cmd
    env = FixedCommandEnv(1, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg)
    policy = load_policy(env, exp, ckpt)
    obs = env.reset()
    env.robot.set_quat(torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=gs.device), zero_velocity=True)
    n_warmup = int(1.0 / env.dt)  # 止まった姿勢から歩き出すまでの 1 秒は記録しない
    qpos = []
    with torch.no_grad():
        for i in range(n_warmup + int(seconds / env.dt)):
            obs, _, _, _ = env.step(policy(obs))
            if i >= n_warmup:
                qpos.append(env.robot.get_qpos()[0].cpu().numpy())
    qpos = np.array(qpos)
    qpos[:, :2] -= qpos[0, :2]
    return qpos, env.dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--output", type=str, default=os.path.join(ROOT, "videos", "go2-sprawling_explainer.mp4"))
    args = parser.parse_args()
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    gs.init(backend=gs.gpu, logging_level="warning")

    # --- 歩行の記録 (学習環境を作って歩かせる)
    walks = {}
    for robot_def in ROBOTS:
        for label, cmd, seconds in WALKS:
            walks[(robot_def[0], label)] = record_walk(robot_def, cmd, seconds)
            print(f"recorded {robot_def[0]} {label}")

    # --- 解説用のシーン: 2 台を y 方向に離して置き、それぞれを同じ相対位置のカメラで撮る
    scene = gs.Scene(show_viewer=False, renderer=gs.renderers.Rasterizer())
    scene.add_entity(gs.morphs.Plane())
    robots = [Robot(scene, *r) for r in ROBOTS]
    video = Video(scene, robots, args.output)
    scene.build()
    for robot in robots:
        robot.setup()

    def update():
        # step を進めずに変えた姿勢は、描画の状態を明示的に更新しないとカメラに映らない
        scene.visualizer.update(force=True)

    def dims(robot):
        feet = np.array(robot.feet()) - robot.base()
        width = feet[:, 1].max() - feet[:, 1].min()
        length = feet[:, 0].max() - feet[:, 0].min()
        hip_y = abs(robot.joint_axes("hip")[1][0][1] - robot.base()[1])
        return width, length, hip_y

    for robot in robots:
        robot.pose(robot.default)
    update()
    sizes = [dims(r) for r in robots]

    # 0. タイトル
    video.card((("Go2  →  Go2 Sprawling", 52, True), ("脚の付け方を変えた改造の解説", 30, False)), 3.5)

    # 1. 全体: 基準姿勢のまわりを 1 周する
    captions = [
        [f"脚は胴体の真下。胴体の高さ {r.stand_height * 100:.0f} cm",
         f"足先の幅 {w * 100:.0f} cm × 前後 {l * 100:.0f} cm"]
        for r, (w, l, _) in zip(robots, sizes)
    ]
    captions[1][0] = f"脚を胴体の横へ張り出す。胴体の高さ {robots[1].stand_height * 100:.0f} cm"
    for t, fade in section_frames(9.0):
        angle = math.radians(-60) + 2 * math.pi * t / 9.0
        offset = (1.5 * math.cos(angle), 1.5 * math.sin(angle), 0.75)
        panels = [video.render_panel(r, offset, (0, 0, -0.1)) for r in robots]
        video.frame(panels, "全体", "sprawling = 脚を横に張り出した姿勢 (トカゲのように低く構える)", captions, fade)

    # 2. 改造①: 付け根の関節の軸 (黄色の矢印) と、その関節だけを動かした様子
    draw_axes(scene, robots, "hip", (1.0, 0.85, 0.1, 1.0))
    captions = [["軸: 前後方向", "→ 脚を左右に開く"], ["軸: 鉛直", "→ 脚を水平に前後へ振る"]]
    for t, fade in section_frames(10.0):
        for robot in robots:
            q = robot.default.copy()
            swing = 0.45 * math.sin(2 * math.pi * t / 2.5)
            for i_leg, leg in enumerate(LEGS):
                q[3 * i_leg] = swing if leg.endswith("L") else -swing
            robot.pose(q)
        update()
        panels = [video.render_panel(r, (1.3, -0.9, 0.8), (0, 0, -0.05)) for r in robots]
        video.frame(panels, "改造①  付け根 (hip) の関節の軸", "黄色の矢印が関節の軸。前後方向の軸を、鉛直の軸に変えた", captions, fade)
    scene.clear_debug_objects()

    # 3. 改造②: 付け根の位置 (赤い球) と足先を結ぶ四角形 (緑) を真上から
    for robot in robots:
        robot.pose(robot.default)
    update()
    for robot in robots:
        for pos, _ in robot.joint_axes("hip"):
            # 付け根の関節は部品の中にあるので、真上から見えるように胴体より上に置く
            scene.draw_debug_sphere(pos + np.array([0.0, 0.0, 0.15]), radius=0.025, color=(1.0, 0.2, 0.2, 1.0))
        feet = robot.feet()
        for a, b in ((0, 1), (1, 3), (3, 2), (2, 0)):
            start, end = feet[a].copy(), feet[b].copy()
            start[2] = end[2] = 0.005
            scene.draw_debug_line(start, end, radius=0.006, color=(0.2, 1.0, 0.4, 1.0))
    captions = [
        [f"付け根: 中心から左右 {hy * 100:.2f} cm", f"足先の幅 {w * 100:.0f} cm × 前後 {l * 100:.0f} cm"]
        for (w, l, hy) in sizes
    ]
    for t, fade in section_frames(8.0):
        panels = [video.render_panel(r, (0.0, 0.0, 2.2), (0, 0, -r.stand_height), up=(1.0, 0.0, 0.0)) for r in robots]
        video.frame(
            panels, "改造②  脚の取り付け位置と向き",
            "真上から (画面の上が前)。付け根 (赤) を左右に 10 cm 外へ出し、脚ごと 90° 回して真横へ向けた", captions, fade,
        )
    scene.clear_debug_objects()

    # 3b. 脚を真横へ向けたので、太もも (thigh) の軸 (水色) は前後方向になる
    draw_axes(scene, robots, "thigh", (0.2, 0.85, 1.0, 1.0))
    captions = [["太ももの軸: 左右方向", "→ 脚を前後に振る"], ["太ももの軸: 前後方向", "→ 脚を横へ上げ下げする"]]
    for t, fade in section_frames(9.0):
        for robot in robots:
            q = robot.default.copy()
            q[1::3] += 0.4 * math.sin(2 * math.pi * t / 3.0)
            robot.pose(q)
        update()
        panels = [video.render_panel(r, (1.4, -1.0, 0.45), (0, 0, -0.08)) for r in robots]
        video.frame(panels, "改造②  脚の取り付け位置と向き (続き)", "斜め前から。水色の矢印が太もも (thigh) の関節の軸", captions, fade)
    scene.clear_debug_objects()

    # 4. 改造③: 後ろ脚を前脚の前後反転に。横から見て、ひざ (すねの関節) を曲げ伸ばしする
    draw_axes(scene, robots, "calf", (1.0, 0.4, 0.9, 1.0), length=0.24)
    captions = [["前脚と後ろ脚が同じ形", "(どちらも、ひざが後ろ向き)"], ["後ろ脚は前脚の前後反転", "→ 前後・左右とも対称"]]
    for t, fade in section_frames(10.0):
        for robot in robots:
            q = robot.default.copy()
            q[2::3] += 0.3 * math.sin(2 * math.pi * t / 3.0)
            robot.pose(q)
        update()
        angle = math.radians(-90) + math.radians(40) * math.sin(math.pi * t / 10.0)
        offset = (1.6 * math.cos(angle), 1.6 * math.sin(angle), 0.6)
        panels = [video.render_panel(r, offset, (0, 0, -0.1)) for r in robots]
        video.frame(
            panels, "改造③  後ろ脚を前脚の前後反転に",
            "後ろ脚の太もも・すねを反対側用の部品に替え、可動域も合わせた。桃色の矢印はひざの軸", captions, fade,
        )
    scene.clear_debug_objects()

    # 5. 学習した方策で歩かせた記録を並べて再生する
    for label, cmd, seconds in WALKS:
        records = [walks[(r.name, label)] for r in robots]
        heights = [qpos[:, 2].mean() for qpos, _ in records]
        speeds = [np.linalg.norm(qpos[-1, :2] - qpos[0, :2]) / (len(qpos) * dt) for qpos, dt in records]
        captions = [[f"胴体の高さ 平均 {h * 100:.0f} cm", f"進んだ速さ 平均 {v:.2f} m/s"] for h, v in zip(heights, speeds)]
        centers = [None, None]
        for t, fade in section_frames(seconds):
            panels = []
            for i, (robot, (qpos, dt)) in enumerate(zip(robots, records)):
                i_step = min(int(t / dt), len(qpos) - 1)
                q = torch.tensor(qpos[i_step], dtype=gs.tc_float, device=gs.device)
                q[1] += robot.lane_y
                robot.entity.set_qpos(q)
                base = q[:3].cpu().numpy()
                centers[i] = base if centers[i] is None else 0.9 * centers[i] + 0.1 * base
            update()
            for robot, center in zip(robots, centers):
                panels.append(video.render_panel(robot, (1.4, -1.6, 0.8), (0, 0, -0.1), center=center))
            video.frame(panels, f"学習した歩き方: {label}", "それぞれの不整地用の方策で、平地を歩かせた記録", captions, fade)

    # 6. まとめ
    w0, l0, hy0 = sizes[0]
    w1, l1, hy1 = sizes[1]
    video.card(
        (
            ("まとめ", 40, True),
            ("①  付け根の関節の軸: 前後方向 → 鉛直", 28, False),
            (f"②  付け根を外へ ({hy0 * 100:.1f} → {hy1 * 100:.1f} cm)、脚を真横へ向けた", 28, False),
            ("③  後ろ脚を前脚の前後反転にして、前後・左右とも対称に", 28, False),
            (f"胴体の高さ {robots[0].stand_height * 100:.0f} → {robots[1].stand_height * 100:.0f} cm、"
             f"足先の幅 {w0 * 100:.0f} → {w1 * 100:.0f} cm", 24, False),
        ),
        6.0,
    )

    video.close()
    print(f"saved {args.output} ({video.n_frames / FPS:.1f} s)")


if __name__ == "__main__":
    main()
