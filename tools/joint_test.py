"""URDF のロボットの各関節を GUI のスライダーで動かして確かめる。

スライダーは PD 制御の目標角 (学習環境と同じ kp/kd とトルク上限で関節が追従する)。
各スライダー右の「sweep」をオンにすると、その関節を可動域の中で往復させ続ける。
画面左の Genesis 標準パネルの Visualization -> Link Frame で各リンクの座標軸も表示できる。

    python tools/joint_test.py                         # Go2 Sprawling の URDF を空中に固定して表示
    python tools/joint_test.py --free                  # 地面に立たせる (重力・接触込み)
    python tools/joint_test.py --urdf go2_sprawling/urdf/go2_orig.urdf
"""

import argparse
import math
import os
import sys
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "go2_rough"))

import genesis as gs
from genesis.ext.pyrender.overlay import ImGuiOverlayPlugin

from train import get_cfgs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--urdf", type=str, default=os.path.join(ROOT, "go2_sprawling", "urdf", "go2.urdf"))
    parser.add_argument("--free", action="store_true", help="胴体を固定せず地面に立たせる")
    parser.add_argument("--height", type=float, default=0.6, help="胴体の初期高さ [m]")
    parser.add_argument("--kp", type=float, default=None, help="PD の P ゲイン (省略時は学習設定の値)")
    parser.add_argument("--kd", type=float, default=None, help="PD の D ゲイン (省略時は学習設定の値)")
    parser.add_argument("--sweep-period", type=float, default=3.0, help="sweep の往復周期 [s]")
    args = parser.parse_args()

    env_cfg = get_cfgs()[0]
    kp = env_cfg["kp"] if args.kp is None else args.kp
    kd = env_cfg["kd"] if args.kd is None else args.kd

    gs.init(backend=gs.gpu, logging_level="warning")
    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01),
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.1, -1.9, args.height + 0.4),
            camera_lookat=(0.0, 0.0, args.height - 0.2),
            camera_fov=40,
            res=(1600, 1000),
            enable_gui=True,
        ),
        show_viewer=True,
    )
    scene.add_entity(gs.morphs.Plane())
    robot = scene.add_entity(
        gs.morphs.URDF(file=args.urdf, pos=(0.0, 0.0, args.height), fixed=not args.free),
        name="robot",
    )
    scene.build()

    joints = [j for j in robot.joints if j.n_dofs == 1]
    names = [j.name for j in joints]
    dofs_idx = [j.dof_start for j in joints]
    lower, upper = (np.asarray(x.cpu()) for x in robot.get_dofs_limit(dofs_idx))
    lower, upper = np.clip(lower, -math.pi, math.pi), np.clip(upper, -math.pi, math.pi)
    force_limit = np.asarray(robot.get_dofs_force_range(dofs_idx)[1].cpu())

    default_pose = np.clip(
        [env_cfg["default_joint_angles"].get(n, 0.0) for n in names], lower, upper
    )
    robot.set_dofs_kp([kp] * len(dofs_idx), dofs_idx)
    robot.set_dofs_kv([kd] * len(dofs_idx), dofs_idx)
    robot.set_dofs_position(default_pose, dofs_idx)

    # GUI (ビューアのスレッド) とメインループで共有する状態
    target = default_pose.copy()
    is_sweeping = np.zeros(len(names), dtype=bool)
    measured_pos = default_pose.copy()
    measured_force = np.zeros(len(names))

    def panel(imgui):
        # overlay として登録したパネルはウィンドウを自分で開く。初回だけ画面右端に置き、以後はドラッグで移動できる
        imgui.set_next_window_pos((imgui.get_io().display_size.x - 620, 60), imgui.Cond_.first_use_ever)
        imgui.begin("Joint test", flags=imgui.WindowFlags_.always_auto_resize)
        imgui.text(f"kp={kp:g}, kd={kd:g}, {'free' if args.free else 'fixed base'}")
        if imgui.button("Default pose"):
            target[:] = default_pose
            is_sweeping[:] = False
        imgui.same_line()
        if imgui.button("All zero"):
            target[:] = np.clip(0.0, lower, upper)
            is_sweeping[:] = False
        imgui.same_line()
        if imgui.button("Stop sweep"):
            is_sweeping[:] = False
        for i, name in enumerate(names):
            if i % 3 == 0:
                imgui.separator()
            changed, value = imgui.slider_float(
                f"##{name}", math.degrees(target[i]), math.degrees(lower[i]), math.degrees(upper[i]), "%.1f deg"
            )
            if changed:
                target[i] = math.radians(value)
                is_sweeping[i] = False
            imgui.same_line()
            _, is_sweeping[i] = imgui.checkbox(f"sweep##{name}", bool(is_sweeping[i]))
            imgui.text(
                f"  {name}: now {math.degrees(measured_pos[i]):6.1f} deg, "
                f"torque {measured_force[i]:+6.2f} / {force_limit[i]:.1f} Nm"
            )
        imgui.end()

    overlay = next(p for p in scene.viewer.plugins if isinstance(p, ImGuiOverlayPlugin))
    # 標準パネルの下に足すとスクロールが必要になるので、独立した浮動ウィンドウにする
    overlay.register_panel(panel, section="overlay")
    print(__doc__)

    start = time.time()
    while scene.viewer.is_alive():
        phase = math.sin(2.0 * math.pi * (time.time() - start) / args.sweep_period)
        mid, half = (lower + upper) / 2.0, (upper - lower) / 2.0
        target[is_sweeping] = (mid + 0.9 * half * phase)[is_sweeping]
        robot.control_dofs_position(target, dofs_idx)
        scene.step()
        measured_pos[:] = np.asarray(robot.get_dofs_position(dofs_idx).cpu())
        measured_force[:] = np.asarray(robot.get_dofs_control_force(dofs_idx).cpu())


if __name__ == "__main__":
    main()
