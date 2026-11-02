"""決め打ちの動作でロボットが跳べるかを確かめる (強化学習なし)。

基準姿勢で立つ -> 0.5 秒かけてしゃがむ -> 目標角を一気に「脚を伸ばした姿勢」に切り替える -> 0.3 秒後に基準姿勢へ戻す。
関節は PD 制御で、トルクは URDF の effort で頭打ちになる (Genesis は URDF の velocity 上限を使わない)。
胴体の上昇量・足先の最高高さ・滞空時間を表示し、--video で横から見た動画を videos/ に保存する。

    python tools/jump_test.py                       # 学習と同じ kp=20
    python tools/jump_test.py --kp 40 --video
    python tools/jump_test.py --extend 0.5 -0.84    # 伸ばした姿勢 (太もも, すね) [rad]
"""

import argparse
import math
import os
import subprocess
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "go2_rough"))

import genesis as gs
from genesis.utils.geom import transform_by_quat

from train import get_cfgs

LEGS = ("FR", "FL", "RR", "RL")
# calf リンク座標での足先 (球の中心) の位置と球の半径
FOOT_OFFSET = (0.0, 0.0, -0.213)
FOOT_RADIUS = 0.022


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=os.path.join(ROOT, "go2_sprawling", "urdf", "go2.urdf"))
    parser.add_argument("--kp", type=float, default=20.0)
    parser.add_argument("--kd", type=float, default=None, help="省略時は kp に合わせて 0.5 * sqrt(kp / 20)")
    parser.add_argument("--crouch", type=float, nargs=2, default=[1.9, -2.6], metavar=("THIGH", "CALF"))
    parser.add_argument("--extend", type=float, nargs=2, default=[0.5, -0.84], metavar=("THIGH", "CALF"))
    parser.add_argument("--video", action="store_true", help="横から見た動画を videos/jump_kp<kp>.mp4 に保存 (0.5 倍速)")
    args = parser.parse_args()
    kd = 0.5 * math.sqrt(args.kp / 20.0) if args.kd is None else args.kd

    env_cfg = get_cfgs()[0]
    dt = 0.005
    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(sim_options=gs.options.SimOptions(dt=dt), show_viewer=False)
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(gs.morphs.URDF(file=args.urdf, pos=(0.0, 0.0, env_cfg["base_init_pos"][2])))
    camera = scene.add_camera(res=(960, 640), pos=(0.0, -2.2, 0.5), lookat=(0.0, 0.0, 0.4), fov=45) if args.video else None
    scene.build()

    names = [f"{leg}_{j}_joint" for leg in LEGS for j in ("hip", "thigh", "calf")]
    dofs = [robot.get_joint(n).dofs_idx_local[0] for n in names]
    calf_links = [robot.get_link(f"{leg}_calf").idx_local for leg in LEGS]
    force_limit = robot.get_dofs_force_range(dofs)[1]

    def pose(thigh, calf):
        return torch.tensor([0.0 if "hip" in n else thigh if "thigh" in n else calf for n in names], device=gs.device)

    default = torch.tensor([env_cfg["default_joint_angles"][n] for n in names], device=gs.device)
    crouch, extend = pose(*args.crouch), pose(*args.extend)
    robot.set_dofs_kp([args.kp] * 12, dofs)
    robot.set_dofs_kv([kd] * 12, dofs)
    robot.set_dofs_position(default, dofs)

    t_crouch, t_jump, t_end = 1.0, 1.8, 3.3
    frames, stand_z = [], None
    max_z, max_clear, flight, longest_flight, max_vel, max_torque_ratio = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for i in range(int(t_end / dt)):
        t = i * dt
        if t < t_crouch:
            target = default
        elif t < t_crouch + 0.5:
            a = (t - t_crouch) / 0.5
            target = (1 - a) * default + a * crouch
        elif t < t_jump:
            target = crouch
        elif t < t_jump + 0.3:
            target = extend
        else:
            target = default
        robot.control_dofs_position(target, dofs)
        scene.step()

        z = robot.get_pos()[2].item()
        if stand_z is None and t >= t_crouch:
            stand_z = z
        if t >= t_jump:
            pos, quat = robot.get_links_pos()[calf_links], robot.get_links_quat()[calf_links]
            foot = pos + transform_by_quat(torch.tensor(FOOT_OFFSET, device=gs.device).expand(4, 3), quat)
            contact = torch.norm(robot.get_links_net_contact_force()[calf_links], dim=-1) > 1.0
            flight = 0.0 if contact.any() else flight + dt
            longest_flight = max(longest_flight, flight)
            max_z = max(max_z, z)
            max_clear = max(max_clear, foot[:, 2].min().item() - FOOT_RADIUS)
            max_vel = max(max_vel, robot.get_dofs_velocity(dofs).abs().max().item())
            max_torque_ratio = max(max_torque_ratio, (robot.get_dofs_control_force(dofs).abs() / force_limit).max().item())
        if camera is not None and i % 4 == 0:
            frames.append(camera.render()[0])

    print(f"kp={args.kp:g} kd={kd:.2f}  crouch={args.crouch}  extend={args.extend}")
    print(f"  立った高さ            {stand_z:.3f} m")
    print(f"  胴体の上昇量          {100 * (max_z - stand_z):.1f} cm")
    print(f"  足先の最高高さ (4 本の最低) {100 * max_clear:.1f} cm")
    print(f"  最長の滞空時間        {longest_flight:.2f} s")
    print(f"  関節速度の最大        {max_vel:.1f} rad/s (URDF の上限: hip/thigh 30.1, calf 20.07。Genesis は未使用)")
    print(f"  トルク / 上限 の最大  {max_torque_ratio:.2f}")

    if frames:
        os.makedirs(os.path.join(ROOT, "videos"), exist_ok=True)
        path = os.path.join(ROOT, "videos", f"jump_kp{args.kp:g}.mp4")
        h, w = frames[0].shape[:2]
        # 描画は 50 fps 相当、25 fps で書き出して 0.5 倍速にする
        ffmpeg = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", "25",
             "-i", "-", "-pix_fmt", "yuv420p", "-vcodec", "libx264", path],
            stdin=subprocess.PIPE,
        )
        for frame in frames:
            ffmpeg.stdin.write(frame.tobytes())
        ffmpeg.stdin.close()
        ffmpeg.wait()
        print(f"  動画: {path}")


if __name__ == "__main__":
    main()
