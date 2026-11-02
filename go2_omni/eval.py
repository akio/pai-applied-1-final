"""学習済みの全方向歩行ポリシーに決まった速度コマンドを順に与えて評価する。

デフォルトはヘッドレスで、コマンドごとの追従誤差と転倒回数を表示する。
--viewer を付けると 1 体をビューアに表示し、コマンドを一定時間ごとに切り替え続ける (Ctrl+C で終了)。
"""

import argparse
import itertools
import os
import pickle
import sys

import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Genesis", "examples", "locomotion"))

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env

# (名前, [前後 m/s, 横 m/s, 旋回 rad/s])
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


class FixedCommandGo2Env(Go2Env):
    """ランダムに再サンプルする代わりに fixed_command を全環境に与える Go2Env。"""

    fixed_command = None

    def _resample_commands(self, envs_idx):
        if self.fixed_command is None:
            self.commands.zero_()
        else:
            self.commands[:] = self.fixed_command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-omni")
    parser.add_argument("--ckpt", type=int, required=True, help="読み込むチェックポイントの iteration")
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--duration", type=float, default=5.0, help="1 コマンドあたりの秒数")
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg = pickle.load(f)
    reward_cfg["reward_scales"] = {}
    # 評価中のエピソード打ち切りを防ぐ
    env_cfg["episode_length_s"] = 1e6

    gs.init(backend=gs.gpu, logging_level="warning")
    env = FixedCommandGo2Env(
        num_envs=1 if args.viewer else 64,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        show_viewer=args.viewer,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{args.ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)

    n_steps = int(args.duration / env.dt)
    # 最初の 1 秒は加速中なので追従誤差の計測から外す
    n_warmup = int(1.0 / env.dt)
    scenarios = itertools.cycle(SCENARIOS) if args.viewer else SCENARIOS

    if not args.viewer:
        print(f"{'command':<14}{'cmd (vx, vy, wz)':>22}{'measured (vx, vy, wz)':>26}{'falls':>7}")
    obs = env.reset()
    with torch.no_grad():
        for name, command in scenarios:
            env.fixed_command = torch.tensor(command, dtype=gs.tc_float, device=gs.device)
            env.commands[:] = env.fixed_command
            measured = torch.zeros(3, device=gs.device)
            n_falls = 0
            for i in range(n_steps):
                obs, _, dones, _ = env.step(policy(obs))
                n_falls += int(dones.sum())
                if i >= n_warmup:
                    measured[:2] += env.base_lin_vel[:, :2].mean(dim=0)
                    measured[2] += env.base_ang_vel[:, 2].mean()
            measured /= n_steps - n_warmup
            if args.viewer:
                print(f"{name}: cmd={command} measured={[round(v, 2) for v in measured.tolist()]}")
            else:
                cmd_str = "(" + ", ".join(f"{v:+.2f}" for v in command) + ")"
                meas_str = "(" + ", ".join(f"{v:+.2f}" for v in measured.tolist()) + ")"
                print(f"{name:<14}{cmd_str:>22}{meas_str:>26}{n_falls:>7}")


if __name__ == "__main__":
    main()
