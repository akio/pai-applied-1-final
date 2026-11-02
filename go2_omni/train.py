"""Go2 の全方向歩行 (前後・横移動・旋回) を学習する。

Genesis/examples/locomotion の Go2Env と設定を流用し、速度コマンドの範囲と学習回数だけを上書きする。
"""

import argparse
import os
import pickle
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "Genesis", "examples", "locomotion"))

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from go2_env import Go2Env
from go2_train import get_cfgs, get_train_cfg

# 速度コマンドの範囲 (ロボット座標系): 前後 [m/s], 横 [m/s], 旋回 [rad/s]
COMMAND_RANGES = {
    "lin_vel_x_range": [-1.0, 1.0],
    "lin_vel_y_range": [-0.6, 0.6],
    "ang_vel_range": [-1.0, 1.0],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-omni")
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    env_cfg, obs_cfg, reward_cfg, command_cfg = get_cfgs()
    command_cfg.update(COMMAND_RANGES)
    train_cfg = get_train_cfg(args.exp_name)

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = Go2Env(
        num_envs=args.num_envs, env_cfg=env_cfg, obs_cfg=obs_cfg, reward_cfg=reward_cfg, command_cfg=command_cfg
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
