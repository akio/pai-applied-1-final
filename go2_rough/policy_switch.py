"""歩行の方策とジャンプの方策を切り替えて動かす。

ふだんは歩行の方策で動かし、ジャンプを頼まれたら次の順に進めて歩行の方策に戻す。
  1. 待機 (settle): 速度指令を 0 にして歩行の方策で立ち止まらせる (止まった状態が min_still_s 続くか、
     max_settle_s 経てば次へ)
  2. ジャンプ (jump): 環境にジャンプ指令を出し、ジャンプの方策に任せる。着地から return_delay_s 経つか
     ジャンプ指令の時間切れで歩行の方策に戻す
どちらの方策も同じロボット・関節の並び・行動のスケールで学習している必要がある。
ジャンプ指令の観測 (2 項目) を持たない歩行の方策には、その 2 項目を除いた観測を渡す。
"""

import os
import pickle

import torch
from tensordict import TensorDict

from rsl_rl.runners import OnPolicyRunner

import genesis as gs

WALK, SETTLE, JUMP = 0, 1, 2
# 観測の並び (env.py の _update_observation): 角速度 3, 重力 3, 指令 3, ジャンプ指令 (num_jump_obs), ...
JUMP_OBS_START = 9


def drop_jump_obs(obs, num_jump_obs):
    """観測からジャンプ指令の項目を除く (ジャンプを学習していない方策に渡す用)。"""
    policy = obs["policy"]
    policy = torch.cat((policy[:, :JUMP_OBS_START], policy[:, JUMP_OBS_START + num_jump_obs :]), dim=1)
    return TensorDict({**dict(obs.items()), "policy": policy}, batch_size=obs.batch_size)


class _NoJumpObsEnv:
    """OnPolicyRunner がモデルの入力の大きさを決めるときに見せる、ジャンプ指令の観測を除いた環境。"""

    def __init__(self, env):
        self.env = env
        self.num_envs, self.num_actions, self.cfg = env.num_envs, env.num_actions, env.cfg

    def get_observations(self):
        return drop_jump_obs(self.env.get_observations(), self.env.num_jump_obs)


def load_policy(env, exp_name, ckpt):
    """logs/<exp_name>/model_<ckpt>.pt の方策を、env の観測をそのまま受け取れる関数として返す。"""
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", exp_name)
    with open(os.path.join(log_dir, "cfgs.pkl"), "rb") as f:
        _, _, _, command_cfg, _, train_cfg = pickle.load(f)
    has_jump_obs = command_cfg.get("jump") is not None
    runner = OnPolicyRunner(env if has_jump_obs else _NoJumpObsEnv(env), train_cfg, log_dir, device=gs.device)
    runner.load(os.path.join(log_dir, f"model_{ckpt}.pt"))
    policy = runner.get_inference_policy(device=gs.device)
    if has_jump_obs or env.num_jump_obs == 0:
        return policy
    return lambda obs: policy(drop_jump_obs(obs, env.num_jump_obs))


class PolicySwitcher:
    """環境ごとに歩行 / 待機 / ジャンプの段階を持ち、段階に応じて方策の行動を選ぶ。"""

    def __init__(
        self, env, walk_policy, jump_policy, settle_speed=0.1, min_still_s=0.3, max_settle_s=1.5, return_delay_s=0.5
    ):
        self.env = env
        self.walk_policy, self.jump_policy = walk_policy, jump_policy
        self.settle_speed, self.min_still_s, self.max_settle_s = settle_speed, min_still_s, max_settle_s
        self.return_delay_s = return_delay_s
        self.mode = torch.full((env.num_envs,), WALK, dtype=torch.long, device=gs.device)
        # 待機中は待機を始めてからの時間、ジャンプ中は着地してからの時間
        self.timer = torch.zeros(env.num_envs, dtype=gs.tc_float, device=gs.device)
        # 待機中に、止まっている状態 (歩行の方策は足踏みは続ける) が続いている時間
        self.still_time = torch.zeros(env.num_envs, dtype=gs.tc_float, device=gs.device)

    @property
    def is_busy(self):
        """待機中かジャンプ中 (速度指令を 0 にすべき環境)。"""
        return self.mode != WALK

    def request_jump(self, env_ids):
        is_walking = self.mode[env_ids] == WALK
        env_ids = env_ids[is_walking]
        self.mode[env_ids] = SETTLE
        self.timer[env_ids] = 0.0
        self.still_time[env_ids] = 0.0

    def reset(self):
        """全環境を歩行に戻す (env.reset() の後に呼ぶ)。"""
        self.mode[:] = WALK

    def act(self, obs):
        env = self.env
        # 待機 -> ジャンプ: 歩行の途中 (脚を振り出している最中) で渡すと、ジャンプの方策が踏み切りで崩れやすい
        is_ready = (self.mode == SETTLE) & ((self.still_time >= self.min_still_s) | (self.timer >= self.max_settle_s))
        if is_ready.any():
            ids = is_ready.nonzero(as_tuple=False)[:, 0]
            env.trigger_jump(ids)
            self.mode[ids] = JUMP
            self.timer[ids] = 0.0
            # trigger_jump が変えた指令とジャンプ指令の観測を、この step の方策の入力にも反映する
            env._update_observation()
            obs = env.get_observations()
        # ジャンプ -> 歩行: 着地から return_delay_s 経つか、ジャンプ指令が時間切れになったら戻す
        is_landed = (self.mode == JUMP) & ~env.is_jump_pending
        is_done = (is_landed & (self.timer >= self.return_delay_s)) | ((self.mode == JUMP) & (env.jump_timer < 0))
        if is_done.any():
            ids = is_done.nonzero(as_tuple=False)[:, 0]
            self.mode[ids] = WALK
            # 環境のジャンプ指令も終わらせ、すぐ速度指令を受け付けるようにする
            env.jump_timer[ids] = -1.0
            env.is_jump_pending[ids] = False
            env._update_observation()
            obs = env.get_observations()

        is_jump = self.mode == JUMP
        actions = self.walk_policy(obs)
        if is_jump.any():
            actions = torch.where(is_jump[:, None], self.jump_policy(obs), actions)
        return actions

    def after_step(self, dones):
        """env.step の後に呼ぶ。段階の経過時間を進め、リセットされた環境は歩行に戻す。"""
        env = self.env
        # ジャンプ中は着地してからの時間を数える (着地前は 0 のまま)
        is_counting = (self.mode == SETTLE) | ((self.mode == JUMP) & ~env.is_jump_pending)
        self.timer = torch.where(is_counting, self.timer + env.dt, self.timer)
        speed = torch.norm(env.base_lin_vel[:, :2], dim=1)
        is_still = (speed < self.settle_speed) & (env.base_ang_vel[:, 2].abs() < 2 * self.settle_speed)
        self.still_time = torch.where((self.mode == SETTLE) & is_still, self.still_time + env.dt, 0.0)
        self.mode[dones] = WALK
