"""不整地歩行の方策を地形タイプ x 難易度ごとに評価する。

ヘッドレス: 各 (地形タイプ, 難易度) に複数体を置き、前進指令で --duration 秒歩かせて
成功率 (転倒せずに歩き続けた割合) と出発地点からの平均移動距離を表にする。
--viewer: 指定した地形・難易度で 1 体を表示し続ける (Ctrl+C で終了)。カメラは出発地点のマスを向けて固定し、
マウスでロボットを中心に回転・ズームできる。--follow を付けるとカメラがロボットに付いて動く。
"""

import argparse
import math
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import TERRAIN_TYPES, Go2RoughEnv
from flip import Go2FlipEnv
from viewer_camera import RobotOrbitCamera


# --scenarios で与える (名前, [前後 m/s, 横 m/s, 旋回 rad/s])
SCENARIOS = [
    ("stand", [0.0, 0.0, 0.0]),
    ("forward", [0.8, 0.0, 0.0]),
    ("backward", [-0.8, 0.0, 0.0]),
    ("left", [0.0, 0.5, 0.0]),
    ("right", [0.0, -0.5, 0.0]),
    ("turn_left", [0.0, 0.0, 1.0]),
    ("turn_right", [0.0, 0.0, -1.0]),
    ("forward+turn", [0.6, 0.0, 0.8]),
    ("diagonal", [0.5, 0.4, 0.0]),
]


class FixedCommandEnv(Go2RoughEnv):
    """ランダムな指令の代わりに fixed_command を与え続ける Go2RoughEnv。"""

    fixed_command = (0.0, 0.0, 0.0)

    def _resample_commands(self, env_ids):
        self.commands[env_ids] = torch.tensor(self.fixed_command, dtype=gs.tc_float, device=gs.device)

    def _trigger_random_jumps(self):
        # ジャンプも外から trigger_jump を呼んだときだけ
        pass


def evaluate_jump(env, policy, obs, n_jumps=3, interval_s=3.0):
    """静止指令のまま interval_s 秒ごとに全台へジャンプ指令を出し、1 回ごとの結果を表にする。

    COM rise はロボット全体の重心の、指令を受けた時点からの最大上昇量 (脚の縮めかたでは変わらない、体ごとの跳躍の高さ)。
    """
    FixedCommandEnv.fixed_command = (0.0, 0.0, 0.0)
    obs = env.reset()
    n_interval = int(interval_s / env.dt)
    print(f"{env.num_envs} robots, jump command every {interval_s:.0f} s")
    print(f"{'jump':<6}{'success':>9}{'COM rise cm':>13}{'base rise cm':>14}{'foot clear cm':>15}{'flight s':>10}{'drift m':>9}{'falls':>7}")
    all_ids = torch.arange(env.num_envs, device=gs.device)
    with torch.no_grad():
        for _ in range(n_interval):  # 指令の前に 1 回分立たせる
            obs, _, _, _ = env.step(policy(obs))
        for k in range(n_jumps):
            env.trigger_jump(all_ids)
            start_xy = env.base_pos[:, :2].clone()
            stand_z = env.base_pos[:, 2] - env.measured_heights.mean(dim=1)
            rise = torch.zeros(env.num_envs, device=gs.device)
            com_rise = torch.zeros(env.num_envs, device=gs.device)
            clear = torch.zeros(env.num_envs, device=gs.device)
            flight = torch.zeros(env.num_envs, device=gs.device)
            is_done = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)  # 着地済み or リセット済み
            is_success = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
            is_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
            drift = torch.zeros(env.num_envs, device=gs.device)
            for _ in range(n_interval):
                was_pending = env.is_jump_pending.clone()
                obs, _, dones, extras = env.step(policy(obs))
                active = ~is_done
                height = env.base_pos[:, 2] - env.measured_heights.mean(dim=1)
                rise = torch.where(active, torch.maximum(rise, height - stand_z), rise)
                com_height = env.com_pos[:, 2] - env.measured_heights.mean(dim=1) - env.jump_start_com_height
                com_rise = torch.where(active, torch.maximum(com_rise, com_height), com_rise)
                feet_z = env.feet_pos[..., 2].min(dim=1).values - env.measured_heights.mean(dim=1)
                is_airborne = ~env.feet_contact.any(dim=1)
                clear = torch.where(active & is_airborne, torch.maximum(clear, feet_z), clear)
                flight = torch.where(active, torch.maximum(flight, env.jump_flight_time), flight)
                landed = active & was_pending & ~env.is_jump_pending & ~dones
                is_success |= landed
                drift = torch.where(landed, torch.norm(env.base_pos[:, :2] - start_xy, dim=1), drift)
                is_fallen |= ~is_done & dones & (extras["time_outs"] < 0.5)
                is_done |= landed | dones
                # 着地後も interval の終わりまでに転んだら転倒に数える
                is_fallen |= is_success & dones & (extras["time_outs"] < 0.5)
            s = is_success
            print(
                f"{k + 1:<6}{s.float().mean() * 100:8.0f}%{com_rise[s].mean() * 100 if s.any() else 0:13.1f}"
                f"{rise[s].mean() * 100 if s.any() else 0:14.1f}"
                f"{(clear[s].mean() - 0.022) * 100 if s.any() else 0:15.1f}{flight[s].mean() if s.any() else 0:10.2f}"
                f"{drift[s].mean() if s.any() else 0:9.2f}{int(is_fallen.sum()):>7}"
            )


def evaluate_flip(env, policy):
    """全台で 1 エピソードずつ宙返りさせ、回転数・成功率などを表にする (train.py --flip のモデル用)。

    成功: 0.75〜1.25 回転して、指令から 2 秒後に胴体が直立 (傾き 25° 以内) し、胴体が一度も地面に触れていない。
    """
    obs = env.reset()
    n = env.num_envs
    has_base_contact = torch.zeros(n, dtype=torch.bool, device=gs.device)
    com_rise = torch.zeros(n, device=gs.device)
    max_roll = torch.zeros(n, device=gs.device)
    start_height = torch.zeros(n, device=gs.device)
    is_upright = torch.zeros(n, dtype=torch.bool, device=gs.device)
    rotation = torch.zeros(n, device=gs.device)
    window = env.jump_cfg["window_s"]
    with torch.no_grad():
        for _ in range(env.max_episode_length - 1):
            obs, _, _, _ = env.step(policy(obs))
            t = env.flip_time
            is_active = (t >= 0.0) & (t < window)
            start_height = torch.where(t < 0.0, env._com_height(), start_height)
            com_rise = torch.where(is_active, torch.maximum(com_rise, env._com_height() - start_height), com_rise)
            has_base_contact |= is_active & (torch.norm(env.contact_forces[:, env.base_link_idx], dim=-1) > 1.0)
            max_roll = torch.where(is_active, torch.maximum(max_roll, env.projected_gravity[:, 1].abs()), max_roll)
            is_end = (t >= window) & (t < window + env.dt)
            is_upright = torch.where(is_end, env.projected_gravity[:, 2] < -math.cos(math.radians(25)), is_upright)
            rotation = torch.where(is_end, env.flip_rotation / (2.0 * math.pi), rotation)
    is_success = (rotation > 0.75) & (rotation < 1.25) & is_upright & ~has_base_contact
    print(f"{n} robots, one back flip each")
    print(f"success (0.75-1.25 turns, upright after {window:.0f} s, no base contact): {is_success.float().mean() * 100:.0f}%")
    print(f"turns: mean {rotation.mean():.2f}, min {rotation.min():.2f}, max {rotation.max():.2f}")
    print(f"upright at end: {is_upright.float().mean() * 100:.0f}%, base contact: {has_base_contact.float().mean() * 100:.0f}%")
    print(f"COM rise: {com_rise.mean() * 100:.1f} cm, max sideways tilt (gravity y): {max_roll.mean():.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-rough")
    parser.add_argument("--ckpt", type=int, required=True)
    parser.add_argument("--cmd", type=float, nargs=3, default=[0.6, 0.0, 0.0], metavar=("VX", "VY", "WZ"))
    parser.add_argument("--duration", type=float, default=5.0, help="ヘッドレス評価で歩かせる秒数")
    parser.add_argument("--per-cell", type=int, default=16, help="ヘッドレス評価で 1 マスあたりに置く台数")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument(
        "--scenarios", action="store_true", help="地形ごとの評価の代わりに、決まった速度指令ごとの追従を測る (平地向け)"
    )
    parser.add_argument("--jump", action="store_true", help="その場ジャンプの指令に対する成功率・高さ・転倒を測る (平地向け)")
    parser.add_argument("--flip", action="store_true", help="宙返りの回転数・成功率を測る (train.py --flip のモデル用)")
    parser.add_argument("--terrain", choices=TERRAIN_TYPES, default="stairs_up", help="--viewer で表示する地形")
    parser.add_argument("--level", type=int, default=4, help="--viewer で表示する難易度")
    parser.add_argument("--follow", action="store_true", help="--viewer のカメラをロボットに付いて動かす")
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    terrain_cfg["curriculum"] = False
    command_cfg["curriculum"] = False
    n_levels, n_types = terrain_cfg["n_levels"], len(TERRAIN_TYPES)

    gs.init(backend=gs.gpu, logging_level="warning")
    FixedCommandEnv.fixed_command = tuple(args.cmd)
    # 宙返りのモデルは、エピソードごとに 1 回宙返りの指令を出す学習時の環境で動かす
    env_class = Go2FlipEnv if "flip" in command_cfg else FixedCommandEnv
    env = env_class(
        num_envs=1 if args.viewer else n_levels * n_types * args.per_cell,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
        show_viewer=args.viewer,
    )
    if args.viewer:
        env.terrain_types[:] = TERRAIN_TYPES.index(args.terrain)
        env.terrain_levels[:] = args.level
    else:
        cell = torch.arange(env.num_envs, device=gs.device) // args.per_cell
        env.terrain_levels[:] = cell // n_types
        env.terrain_types[:] = cell % n_types

    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    obs = env.reset()

    if args.viewer:
        # 既定のカメラはワールド原点 (地形の隅) を向いているので、出発地点のマスを斜め上から見下ろす位置に置く
        center = env.env_origins[0].cpu().numpy()
        distance = 1.5 if args.follow else 4.0
        env.scene.viewer.set_camera_pose(pos=center + (-distance, -distance, distance), lookat=center)
        # マウス操作の中心をロボットに置く (--follow ならカメラもロボットに付いて平行移動する)
        orbit = RobotOrbitCamera(env.scene.viewer)
        with torch.no_grad():
            while True:
                obs, _, _, _ = env.step(policy(obs))
                orbit.update(env.base_pos[0].cpu().numpy(), follow=args.follow)

    if args.jump:
        evaluate_jump(env, policy, obs)
        return
    if args.flip:
        evaluate_flip(env, policy)
        return

    if args.scenarios:
        print(f"{'command':<14}{'cmd (vx, vy, wz)':>22}{'measured (vx, vy, wz)':>26}{'falls':>7}")
        n_steps, n_warmup = int(args.duration / env.dt), int(1.0 / env.dt)
        with torch.no_grad():
            for name, command in SCENARIOS:
                FixedCommandEnv.fixed_command = tuple(command)
                obs = env.reset()
                measured = torch.zeros(3, device=gs.device)
                n_falls = 0
                for i in range(n_steps):
                    obs, _, dones, extras = env.step(policy(obs))
                    n_falls += int((dones & (extras["time_outs"] < 0.5)).sum())
                    if i >= n_warmup:
                        measured[:2] += env.base_lin_vel[:, :2].mean(dim=0)
                        measured[2] += env.base_ang_vel[:, 2].mean()
                measured /= n_steps - n_warmup
                cmd_str = "(" + ", ".join(f"{v:+.2f}" for v in command) + ")"
                meas_str = "(" + ", ".join(f"{v:+.2f}" for v in measured.tolist()) + ")"
                print(f"{name:<14}{cmd_str:>22}{meas_str:>26}{n_falls:>7}")
        return

    origins = env.env_origins[:, :2].clone()
    distance = torch.zeros(env.num_envs, device=gs.device)
    is_fallen = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
    # 転倒または地形の端への到達でリセットされた環境は、その時点で計測を打ち切る
    is_finished = torch.zeros(env.num_envs, dtype=torch.bool, device=gs.device)
    with torch.no_grad():
        for _ in range(int(args.duration / env.dt)):
            pos_before = env.base_pos[:, :2].clone()
            obs, _, dones, extras = env.step(policy(obs))
            is_active = ~is_finished
            # dones の環境はリセット後の位置になっているので、直前の位置で距離を測る
            pos = torch.where(dones[:, None], pos_before, env.base_pos[:, :2])
            distance = torch.where(is_active, torch.norm(pos - origins, dim=1), distance)
            is_fallen |= is_active & dones & (extras["time_outs"] < 0.5)
            is_finished |= dones

    success = (~is_fallen).float().view(n_levels, n_types, args.per_cell).mean(dim=2)
    mean_dist = distance.view(n_levels, n_types, args.per_cell).mean(dim=2)
    print(f"command {args.cmd}, {args.duration:.0f} s, {args.per_cell} robots per cell")
    print("success rate / mean distance [m]")
    print(f"{'level':<6}" + "".join(f"{t:>14}" for t in TERRAIN_TYPES))
    for level in range(n_levels):
        cells = [f"{success[level, t] * 100:4.0f}% {mean_dist[level, t]:4.1f}m" for t in range(n_types)]
        print(f"{level:<6}" + "".join(f"{c:>14}" for c in cells))


if __name__ == "__main__":
    main()
