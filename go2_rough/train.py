"""不整地を全方向に歩く Go2 の方策を学習する。"""

import argparse
import os
import pickle
import shutil
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

from env import Go2RoughEnv
from flip import Go2FlipEnv, add_flip_cfgs


def get_train_cfg(exp_name, use_symmetry=True):
    return {
        "algorithm": {
            "class_name": "PPO",
            "clip_param": 0.2,
            "desired_kl": 0.01,
            "entropy_coef": 0.01,
            "gamma": 0.99,
            "lam": 0.95,
            "learning_rate": 0.001,
            "max_grad_norm": 1.0,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "schedule": "adaptive",
            "use_clipped_value_loss": True,
            "value_loss_coef": 1.0,
            # 左右・前後に鏡映した経験も学習に使い、どちら向きにも同じ歩き方をさせる (go2_rough/symmetry.py)
            "symmetry_cfg": {
                "data_augmentation_func": "symmetry:augment",
                "use_data_augmentation": True,
                "use_mirror_loss": False,
            }
            if use_symmetry
            else None,
        },
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "distribution_cfg": {"class_name": "GaussianDistribution", "init_std": 1.0, "std_type": "scalar"},
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
        },
        # critic には胴体の並進速度 (実機では直接測れない量) も渡す
        "obs_groups": {
            "actor": ["policy"],
            "critic": ["policy", "privileged"],
        },
        "num_steps_per_env": 24,
        "save_interval": 100,
        "run_name": exp_name,
        "logger": "tensorboard",
    }


def get_cfgs():
    env_cfg = {
        # ロボットのモデル。Genesis 同梱の Go2 の脚の付き方を変えた改造版 Go2 Sprawling (go2_sprawling/) を使う
        "urdf_file": os.path.join(ROOT, "go2_sprawling", "urdf", "go2.urdf"),
        "num_actions": 12,
        "default_joint_angles": {
            "FL_hip_joint": 0.0,
            "FR_hip_joint": 0.0,
            "RL_hip_joint": 0.0,
            "RR_hip_joint": 0.0,
            "FL_thigh_joint": 1.5706,
            "FR_thigh_joint": 1.5706,
            "RL_thigh_joint": 1.5706,
            "RR_thigh_joint": 1.5706,
            "FL_calf_joint": -1.5706 * 1.2,
            "FR_calf_joint": -1.5706 * 1.2,
            "RL_calf_joint": -1.5706 * 1.2,
            "RR_calf_joint": -1.5706 * 1.2,
        },
        "joint_names": [
            "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
            "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
            "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
            "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        ],
        # モーターの速度上限 [rad/s] (URDF の velocity)。env.py で、上限に近づくほど回る向きのトルクを下げる
        "dof_vel_limits": {"hip": 30.1, "thigh": 30.1, "calf": 20.07},
        "kp": 20.0,
        "kd": 0.5,
        # 胴体の上向きベクトルと鉛直のなす角が 60° を超えたら転倒とみなす (cos 60° = 0.5)
        "termination_min_gravity_z": 0.5,
        "base_init_pos": [0.0, 0.0, 0.25],
        "base_init_quat": [1.0, 0.0, 0.0, 0.0],
        "episode_length_s": 20.0,
        "resampling_time_s": 10.0,
        "action_scale": 0.25,
        "simulate_action_latency": True,
        "clip_actions": 100.0,
    }
    obs_cfg = {
        "obs_scales": {
            "lin_vel": 2.0,
            "ang_vel": 0.25,
            "dof_pos": 1.0,
            "dof_vel": 0.05,
            "height_measurements": 5.0,
        },
        # 高さスキャンの点 (胴体中心から, 胴体の向き基準) [m]: 13 x 9 = 117 点
        "height_scan_x": [round(-0.6 + 0.1 * i, 2) for i in range(13)],
        "height_scan_y": [round(-0.4 + 0.1 * i, 2) for i in range(9)],
    }
    reward_cfg = {
        "tracking_sigma": 0.25,
        "base_height_target": 0.2,
        # 今の歩き方の振り出し時間 (約 0.25 s) に合わせる
        "feet_air_time_target": 0.25,
        "only_positive_rewards": True,
        "reward_scales": {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "lin_vel_z": -1.0,
            "ang_vel_xy": -0.05,
            "base_height": -10.0,
            "action_rate": -0.01,
            "similar_to_default": -0.05,
            "feet_air_time": 1.0,
            "collision": -1.0,
            "feet_slip": -0.25,
            "hip_deviation": -0.2,
        },
    }
    command_cfg = {
        "num_commands": 3,
        # 前後の速度は ±1.0 m/s から始め、追従報酬の平均が curriculum_threshold を超えるたびに
        # curriculum_step ずつ広げて ±2.0 m/s まで上げる (前後の鏡映を使うので範囲は前後対称にする)
        "lin_vel_x_range": [-2.0, 2.0],
        "lin_vel_x_init_range": [-1.0, 1.0],
        "curriculum": True,
        "curriculum_threshold": 0.8,
        "curriculum_step": 0.1,
        "lin_vel_y_range": [-0.6, 0.6],
        "ang_vel_range": [-1.0, 1.0],
        "min_lin_vel": 0.2,
    }
    terrain_cfg = {
        "n_levels": 8,
        "subterrain_size": 6.0,
        "border_size": 3.0,
        "platform_size": 1.5,
        "horizontal_scale": 0.1,
        "vertical_scale": 0.005,
        "max_init_level": 2,
        "curriculum": True,
        "seed": 0,
    }
    return env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg


def add_jump_cfgs(reward_cfg, command_cfg):
    """その場ジャンプの指令と報酬を加える (--jump)。"""
    command_cfg["jump"] = {
        # 指令から window_s 秒の間は速度指令を 0 にする。この間に跳んで着地すれば成功
        "window_s": 2.0,
        # 学習中は平均 interval_s 秒に 1 回、ランダムに指令を出す。最初の start_iteration 回は歩行だけを学習させる
        "interval_s": 6.0,
        "start_iteration": 300,
        # 4 本とも浮いている時間がこれ以上続いてから着地したら、ジャンプ完了とみなす
        "min_flight_s": 0.1,
        # jump_stretch に数える胴体の上昇量の上限 [m]
        "max_height": 0.35,
        # jump_com に数える重心の上昇量の上限 [m] (大きいと、着地で転びやすい無理な高さを狙って歩行を覚えなくなる)
        "max_com_rise": 0.35,
    }
    reward_cfg["jump_masked_rewards"] = ["base_height", "lin_vel_z", "similar_to_default"]
    # 足先の高さではなく重心の高さに報酬を与え、脚を縮めて足先だけ持ち上げる跳び方を避ける。空中では基準姿勢を保たせる
    reward_cfg["reward_scales"].update({"jump_stretch": 1.0, "jump_com": 100.0, "jump_flight_pose": -1.0})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--exp-name", type=str, default="go2-rough")
    parser.add_argument("-b", "--num-envs", type=int, default=4096)
    parser.add_argument("--max-iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--flat", action="store_true", help="不整地の代わりに平らな地面で学習する")
    parser.add_argument("--jump", action="store_true", help="その場ジャンプの指令も学習する")
    parser.add_argument(
        "--flip", action="store_true", help="後方宙返りだけを学習する (平地, flip.py。歩行は学習しない)"
    )
    parser.add_argument(
        "--resume", type=str, default=None, metavar="EXP/model_N.pt",
        help="logs/ 以下の学習済みモデルの重みから学習を続ける (観測・行動の形が同じ設定のみ)",
    )
    parser.add_argument(
        "--strong", action="store_true",
        help="関節のトルク上限 2 倍・速度上限 1.5 倍の増強版モデル (go2_sprawling/urdf/go2_strong.urdf) を使う",
    )
    parser.add_argument("--no-symmetry", action="store_true", help="鏡映した経験による対称性の誘導を使わない")
    args = parser.parse_args()

    log_dir = os.path.join(ROOT, "logs", args.exp_name)
    env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg = get_cfgs()
    if args.flat:
        # 平地ではどの難易度のマスも同じなので、難易度の自動調整は使わない
        terrain_cfg.update(flat=True, curriculum=False)
    if args.jump:
        add_jump_cfgs(reward_cfg, command_cfg)
        if args.resume:
            # 読み込むモデルはもう歩けるので、最初からジャンプ指令を出す
            command_cfg["jump"]["start_iteration"] = 0
    if args.strong:
        env_cfg["urdf_file"] = os.path.join(ROOT, "go2_sprawling", "urdf", "go2_strong.urdf")
        # Genesis は URDF の velocity を使わないので、モーターの速度上限も同じ倍率にする
        env_cfg["dof_vel_limits"] = {name: limit * 1.5 for name, limit in env_cfg["dof_vel_limits"].items()}
    if args.flip:
        terrain_cfg.update(flat=True, curriculum=False)
        add_flip_cfgs(env_cfg, reward_cfg, command_cfg)
    # 宙返りは前後の向きが決まった動きなので、前後の鏡映を含む対称性の誘導は使えない
    train_cfg = get_train_cfg(args.exp_name, use_symmetry=not (args.no_symmetry or args.flip))

    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)
    os.makedirs(log_dir)
    with open(os.path.join(log_dir, "cfgs.pkl"), "wb") as f:
        pickle.dump([env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, train_cfg], f)

    gs.init(backend=gs.gpu, precision="32", logging_level="warning", seed=args.seed, performance_mode=True)
    env = (Go2FlipEnv if args.flip else Go2RoughEnv)(
        num_envs=args.num_envs,
        env_cfg=env_cfg,
        obs_cfg=obs_cfg,
        reward_cfg=reward_cfg,
        command_cfg=command_cfg,
        terrain_cfg=terrain_cfg,
    )
    runner = OnPolicyRunner(env, train_cfg, log_dir, device=gs.device)
    if args.resume:
        runner.load(os.path.join(ROOT, "logs", args.resume))
    runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)


if __name__ == "__main__":
    main()
