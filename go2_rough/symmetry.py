"""rsl_rl の対称性拡張 (PPO の symmetry_cfg) に渡す、観測と行動の鏡映。

Go2 Sprawling は左右 (y -> -y) と前後 (x -> -x) の鏡映で脚どうしが重なる形をしている。
鏡映した状態では、脚を入れ替えたうえで付け根 (hip, 鉛直軸まわり) の角度の符号を反転し、太もも・すねの角度はそのまま使う。
(順運動学で確認済み。胴体の重心が 2 cm 前寄りな点だけは前後の鏡映で厳密には一致しない)

学習データの各サンプルに、左右・前後・両方 (= 180° 回転) の鏡映を加えて 4 倍にし、
どちら向きにも同じ歩き方をするよう方策を誘導する。
"""

import torch
from tensordict import TensorDict

# 鏡映ごとの脚の入れ替え
LEG_SWAPS = {
    "lr": {"FL": "FR", "FR": "FL", "RL": "RR", "RR": "RL"},
    "fb": {"FL": "RL", "RL": "FL", "FR": "RR", "RR": "FR"},
    "rot": {"FL": "RR", "RR": "FL", "FR": "RL", "RL": "FR"},
}
# 胴体座標の極性ベクトル (速度・重力) と軸性ベクトル (角速度) の符号
POLAR_SIGNS = {"lr": (1.0, -1.0, 1.0), "fb": (-1.0, 1.0, 1.0), "rot": (-1.0, -1.0, 1.0)}
AXIAL_SIGNS = {"lr": (-1.0, 1.0, -1.0), "fb": (1.0, -1.0, -1.0), "rot": (-1.0, -1.0, 1.0)}

_cache = {}


def _mirror_maps(env, device):
    """鏡映ごとに (観測の並べ替え添字, 観測の符号, 行動の並べ替え添字, 行動の符号) を作る。"""
    key = (id(env), str(device))
    if key in _cache:
        return _cache[key]

    joint_names = env.env_cfg["joint_names"]
    n_x, n_y = len(env.obs_cfg["height_scan_x"]), len(env.obs_cfg["height_scan_y"])
    maps = {}
    for kind, swap in LEG_SWAPS.items():
        # 行動 (関節ごと): 鏡映後の関節 i には、入れ替え元の関節の値を (hip なら符号反転して) 入れる
        joint_idx = [joint_names.index(swap[name[:2]] + name[2:]) for name in joint_names]
        is_hip = torch.tensor(["hip" in name for name in joint_names])
        # 180° 回転は鏡映を 2 回重ねたものなので、hip の符号は元に戻る
        joint_sign = torch.where(is_hip & (kind != "rot"), -1.0, 1.0)

        # 高さスキャン: 点の並びは x 方向が外側、y 方向が内側 (env.py の meshgrid indexing="ij")
        ix, iy = torch.meshgrid(torch.arange(n_x), torch.arange(n_y), indexing="ij")
        if kind in ("fb", "rot"):
            ix = n_x - 1 - ix
        if kind in ("lr", "rot"):
            iy = n_y - 1 - iy
        scan_idx = (ix * n_y + iy).flatten()

        # 観測 (env.py の _update_observation の並び): 角速度, 重力, 指令, ジャンプ指令, 関節角, 関節速度, 前回の行動, 高さスキャン
        command_sign = (POLAR_SIGNS[kind][0], POLAR_SIGNS[kind][1], AXIAL_SIGNS[kind][2])
        idx, sign, offset = [], [], 0
        for block_idx, block_sign in (
            (torch.arange(3), torch.tensor(AXIAL_SIGNS[kind])),
            (torch.arange(3), torch.tensor(POLAR_SIGNS[kind])),
            (torch.arange(3), torch.tensor(command_sign)),
            # ジャンプ指令 (指令中かどうか, 経過時間) は向きに関係しない
            (torch.arange(env.num_jump_obs), torch.ones(env.num_jump_obs)),
            (torch.tensor(joint_idx), joint_sign),
            (torch.tensor(joint_idx), joint_sign),
            (torch.tensor(joint_idx), joint_sign),
            (scan_idx, torch.ones(len(scan_idx))),
        ):
            idx.append(block_idx + offset)
            sign.append(block_sign)
            offset += len(block_idx)
        maps[kind] = {
            "policy": (torch.cat(idx).to(device), torch.cat(sign).to(device)),
            # privileged は胴体の並進速度 (極性ベクトル) だけ
            "privileged": (torch.arange(3, device=device), torch.tensor(POLAR_SIGNS[kind], device=device)),
            "actions": (torch.tensor(joint_idx, device=device), joint_sign.to(device)),
        }
    _cache[key] = maps
    return maps


def _mirror(x, idx_sign):
    idx, sign = idx_sign
    return x[..., idx] * sign


def augment(env, obs=None, actions=None):
    """観測と行動に左右・前後・180° 回転の鏡映を加え、バッチを 4 倍にする (元のサンプルが先頭)。"""
    # 環境が作る TensorDict は device を持たないので、中身のテンソルから取る
    device = actions.device if actions is not None else obs["policy"].device
    maps = _mirror_maps(env, device)
    kinds = ("lr", "fb", "rot")

    obs_aug = None
    if obs is not None:
        obs_aug = TensorDict(
            {
                key: torch.cat([obs[key]] + [_mirror(obs[key], maps[k][key]) for k in kinds], dim=0)
                for key in obs.keys()
            },
            batch_size=[obs.batch_size[0] * (len(kinds) + 1)],
            device=obs.device,
        )
    actions_aug = None
    if actions is not None:
        actions_aug = torch.cat([actions] + [_mirror(actions, maps[k]["actions"]) for k in kinds], dim=0)
    return obs_aug, actions_aug
