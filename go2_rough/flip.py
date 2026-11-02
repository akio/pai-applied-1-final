"""後方宙返り (バックフリップ) を学習する環境と設定。

各エピソードで立った状態から 1 回だけ宙返りの指令を出し、構え -> 踏み切り -> 回転 -> 着地 -> 立つ までを学習する。
指令はジャンプ指令 (env.py の trigger_jump) と同じ仕組みで、観測も同じ並び (指令中かどうか, 指令からの経過時間) なので、
policy_switch.py でジャンプの方策の代わりに使える。
報酬は指令からの経過時間 t で区切る (ziyanx02/Genesis-backflip の報酬の考え方に合わせている)。
  t < 0.5 s       構え: 足を地面から離さない (しゃがむなどは自由)
  0.5 - 0.75 s    踏み切り: 上向きの速度に報酬
  0.5 - 1.0 s     回転: 後ろ向きの回転速度に報酬。胴体の向きの目標を 0 から 360° まで回す
  t > 1.4 s       着地後: 立ったときの高さに戻す
"""

import math

import torch

import genesis as gs

from env import Go2RoughEnv

# 関節の並び (train.py の joint_names): FR, FL, RR, RL の順に hip, thigh, calf
LEFT_RIGHT_PAIRS = ((0, 3), (6, 9))


def add_flip_cfgs(env_cfg, reward_cfg, command_cfg):
    """train.py の設定を宙返りの学習用に書き換える (--flip)。"""
    flip = command_cfg["flip"] = {
        # 指令を出す時刻 (エピソード開始から) の範囲 [s]。開始直後は落下して着地した姿勢が落ち着かないので避ける
        "trigger_range_s": [0.5, 1.0],
        "prepare_s": 0.5,
        "push_s": 0.25,
        "rotate_s": 0.5,
        "land_s": 1.4,
        # 回転速度の報酬の上限 [rad/s]。0.5 s で 1 回転する平均の速さ (12.6) 近くまで報酬が増えるようにする
        "max_pitch_rate": 12.0,
        "max_up_vel": 3.0,
    }
    # ジャンプ指令の仕組み (指令中の速度指令 0, 観測, 着地の判定) を使う
    command_cfg["jump"] = {"window_s": 2.0, "min_flight_s": 0.1, "is_flip": True}
    command_cfg.update(
        lin_vel_x_range=[0.0, 0.0], lin_vel_y_range=[0.0, 0.0], ang_vel_range=[0.0, 0.0], curriculum=False
    )
    command_cfg.pop("lin_vel_x_init_range", None)
    env_cfg["episode_length_s"] = flip["trigger_range_s"][1] + command_cfg["jump"]["window_s"] + 0.2
    env_cfg["resampling_time_s"] = env_cfg["episode_length_s"]
    env_cfg["terminate_on_fall"] = False
    # 罰が中心の報酬なので、1 ステップの合計を 0 で切り捨てると罰が効かなくなる
    reward_cfg["only_positive_rewards"] = False
    reward_cfg["reward_scales"] = {
        "flip_pitch_rate": 5.0,
        "flip_up_vel": 20.0,
        "flip_orientation": -3.0,
        "flip_feet_on_ground": -30.0,
        "flip_height": -10.0,
        # 弱いと、踏み切りで胴体の前端を地面に押し当てて回転のきっかけにする動きを覚える
        "flip_base_contact": -200.0,
        "roll": -10.0,
        "yaw_rate": -1.0,
        "action_mirror": -0.1,
        "action_rate": -0.001,
    }


class Go2FlipEnv(Go2RoughEnv):
    """立った状態から 1 回だけ後方宙返りの指令を受ける Go2RoughEnv (平地で使う)。"""

    def __init__(self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, **kwargs):
        self.flip_cfg = command_cfg["flip"]
        # 指令を出す時刻 (エピソード開始から) と、指令からの後ろ向きの回転角の累計
        self.flip_trigger_time = torch.zeros(num_envs, dtype=gs.tc_float, device=gs.device)
        self.flip_rotation = torch.zeros(num_envs, dtype=gs.tc_float, device=gs.device)
        super().__init__(num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, **kwargs)

    @property
    def flip_time(self):
        """宙返りの指令からの経過時間 [s] (指令前は負)。"""
        return self.episode_length_buf * self.dt - self.flip_trigger_time

    def _trigger_random_jumps(self):
        t = self.flip_time
        is_triggered = (t >= 0.0) & (t < self.dt) & (self.jump_timer < 0)
        self.trigger_jump(is_triggered.nonzero(as_tuple=False)[:, 0])

    def step(self, actions):
        result = super().step(actions)
        t = self.flip_time
        is_flipping = (t >= 0.0) & (t < self.jump_cfg["window_s"])
        self.flip_rotation += -self.base_ang_vel[:, 1] * self.dt * is_flipping
        return result

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        # 終わったエピソードの回転角 (1 回転 = 1.0) を記録する
        rotation = self.flip_rotation[env_ids] / (2.0 * math.pi)
        super()._reset_idx(env_ids)
        self.extras["episode"]["flip_rotation"] = rotation.mean()
        low, high = self.flip_cfg["trigger_range_s"]
        self.flip_trigger_time[env_ids] = low + (high - low) * torch.rand(len(env_ids), device=gs.device)
        self.flip_rotation[env_ids] = 0.0

    def _flip_window(self, start, end):
        t = self.flip_time
        return (t >= start) & (t < end)

    # ------------------------------------------------------------------ rewards

    def _reward_flip_pitch_rate(self):
        # 後ろ向き (機首上げ) の回転は胴体 y 軸まわりの負の角速度
        f = self.flip_cfg
        pitch_rate = torch.clip(-self.base_ang_vel[:, 1], -f["max_pitch_rate"], f["max_pitch_rate"])
        return pitch_rate * self._flip_window(f["prepare_s"], f["prepare_s"] + f["rotate_s"])

    def _reward_flip_up_vel(self):
        f = self.flip_cfg
        up_vel = torch.clip(self.robot.get_vel()[:, 2], max=f["max_up_vel"])
        return up_vel * self._flip_window(f["prepare_s"], f["prepare_s"] + f["push_s"])

    def _reward_flip_orientation(self):
        # 目標の機首上げ角を回転の時間帯に 0 から 2 pi まで一定の速さで増やし、胴体から見た重力の向きで比べる
        f = self.flip_cfg
        progress = torch.clip((self.flip_time - f["prepare_s"]) / f["rotate_s"], 0.0, 1.0)
        pitch = 2.0 * math.pi * progress
        target = torch.stack((-torch.sin(pitch), torch.zeros_like(pitch), -torch.cos(pitch)), dim=-1)
        return torch.sum(torch.square(self.projected_gravity - target), dim=1)

    def _reward_flip_feet_on_ground(self):
        # 構えの間に足を浮かせた高さ (踏み切りの前に跳ねたり、脚を振り上げたりしないように)
        ground = self._terrain_heights(self.feet_pos[..., :2])
        height = torch.clip(self.feet_pos[..., 2] - ground - 0.02, min=0.0)
        return height.sum(dim=1) * (self.flip_time < self.flip_cfg["prepare_s"])

    def _reward_flip_height(self):
        # 構えの前半と着地の後に、立ったときの胴体の高さからずれた量
        t = self.flip_time
        is_standing = (t < self.flip_cfg["prepare_s"] - 0.1) | (t > self.flip_cfg["land_s"])
        height = self.base_pos[:, 2] - self.measured_heights.mean(dim=1)
        return torch.square(height - self.reward_cfg["base_height_target"]) * is_standing

    def _reward_flip_base_contact(self):
        # 胴体の接地 (深くしゃがんで胴体を地面につける、背中から落ちる)。学習中は接地しても終了しないので罰で避けさせる
        return (torch.norm(self.contact_forces[:, self.base_link_idx], dim=-1) > 1.0).float()

    def _reward_roll(self):
        return torch.square(self.projected_gravity[:, 1])

    def _reward_yaw_rate(self):
        return torch.abs(self.base_ang_vel[:, 2])

    def _reward_action_mirror(self):
        # 左右の脚で鏡に映した動きになっていない量 (左右の鏡映では hip だけ符号が反転する。symmetry.py 参照)
        a = self.actions
        total = torch.zeros(self.num_envs, dtype=gs.tc_float, device=gs.device)
        for right, left in LEFT_RIGHT_PAIRS:
            total += torch.square(a[:, right] + a[:, left])
            total += torch.sum(torch.square(a[:, right + 1 : right + 3] - a[:, left + 1 : left + 3]), dim=1)
        return total
