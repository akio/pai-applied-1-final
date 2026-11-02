"""不整地 (坂・凹凸・階段・障害物) を歩く Go2 の強化学習環境。

Genesis/examples/locomotion/go2_env.py の Go2Env をベースに、次を追加・変更している。
  - 地形: 列 = 地形タイプ, 行 = 難易度 のグリッド状の不整地
  - カリキュラム: 遠くまで歩けたら難易度を上げ、指令の半分も進めなければ下げる
  - 観測: 胴体周辺の地面の高さスキャン (actor/critic) と胴体の並進速度 (critic のみ)
  - 終了条件: 胴体の接地 / 大きな傾き (平面版の roll/pitch 10° は坂で使えない)
  - 報酬: 胴体高さを足元の地面基準に変更、足の滞空時間・太もも衝突・胴体の揺れを追加
"""

import math
import os

import numpy as np
import torch
from tensordict import TensorDict

import genesis as gs
from genesis.ext.isaacgym import terrain_utils
from genesis.utils.geom import inv_quat, quat_to_xyz, transform_by_quat, transform_quat_by_quat

TERRAIN_TYPES = ["slope_up", "slope_down", "rough", "stairs_up", "stairs_down", "obstacles"]


def robot_name(env_cfg):
    """動画のファイル名などに使うロボットの呼び名。"""
    urdf = os.path.basename(env_cfg.get("urdf_file", ""))
    if urdf == "go2_strong.urdf":
        return "go2-sprawling-strong"
    if urdf == "go2.urdf" and "go2_sprawling" in env_cfg["urdf_file"]:
        return "go2-sprawling"
    return "go2"


def gs_rand(lower, upper, batch_shape):
    return (upper - lower) * torch.rand(size=(*batch_shape, *lower.shape), dtype=gs.tc_float, device=gs.device) + lower


def make_height_field(terrain_cfg):
    """難易度 (行, x 方向) x 地形タイプ (列, y 方向) のサブ地形を並べ、外周に平坦な縁を付けた高さマップを返す。

    返り値は vertical_scale 単位の整数高さ。サブ地形 (level, type) の中心はワールド座標で
    (border + (level + 0.5) * size, border + (type + 0.5) * size)。
    """
    hs, vs = terrain_cfg["horizontal_scale"], terrain_cfg["vertical_scale"]
    n_levels, size = terrain_cfg["n_levels"], terrain_cfg["subterrain_size"]
    n_cells = int(size / hs)
    n_border = int(terrain_cfg["border_size"] / hs)
    platform = terrain_cfg["platform_size"]
    height_field = np.zeros(
        (n_levels * n_cells + 2 * n_border, len(TERRAIN_TYPES) * n_cells + 2 * n_border), dtype=gs.np_float
    )
    # 平地: 同じ広さ・同じ配置 (出発地点や地形の端の判定がそのまま使える) の高さ 0 の地面
    if terrain_cfg.get("flat", False):
        return height_field

    rng_state = np.random.get_state()
    np.random.seed(terrain_cfg["seed"])
    for level in range(n_levels):
        difficulty = level / max(n_levels - 1, 1)
        slope = 0.05 + 0.35 * difficulty
        rough_height = 0.02 + 0.06 * difficulty
        step_height = 0.03 + 0.10 * difficulty
        obstacle_height = 0.03 + 0.12 * difficulty
        for i_type, terrain_type in enumerate(TERRAIN_TYPES):
            sub = terrain_utils.SubTerrain(width=n_cells, length=n_cells, vertical_scale=vs, horizontal_scale=hs)
            # 中心 (出発地点) が谷 / 山になるので、"up" は中心から登り、"down" は中心から下る
            if terrain_type == "slope_up":
                terrain_utils.pyramid_sloped_terrain(sub, slope=-slope, platform_size=platform)
            elif terrain_type == "slope_down":
                terrain_utils.pyramid_sloped_terrain(sub, slope=slope, platform_size=platform)
            elif terrain_type == "rough":
                terrain_utils.random_uniform_terrain(
                    sub, min_height=-rough_height, max_height=rough_height, step=vs, downsampled_scale=0.2
                )
            elif terrain_type == "stairs_up":
                terrain_utils.pyramid_stairs_terrain(
                    sub, step_width=0.31, step_height=-step_height, platform_size=platform
                )
            elif terrain_type == "stairs_down":
                terrain_utils.pyramid_stairs_terrain(sub, step_width=0.31, step_height=step_height, platform_size=platform)
            elif terrain_type == "obstacles":
                terrain_utils.discrete_obstacles_terrain(
                    sub, max_height=obstacle_height, min_size=0.4, max_size=1.2, num_rects=30, platform_size=platform
                )
            x0 = n_border + level * n_cells
            y0 = n_border + i_type * n_cells
            height_field[x0 : x0 + n_cells, y0 : y0 + n_cells] = sub.height_field_raw
    np.random.set_state(rng_state)
    return height_field


def make_checker_image(height_field_shape, horizontal_scale, tile_size, px_per_tile=16):
    """地形全体に貼る市松模様の画像 (1 マス約 tile_size [m] 四方) を返す。

    Terrain の UV は地形の端から端までが 0〜1 (u が x 方向, v が y 方向) なので、地形の縦横比に合わせた 1 枚の画像にする。
    """
    n_x, n_y = (max(1, round((n - 1) * horizontal_scale / tile_size)) for n in height_field_shape)
    iy, ix = np.meshgrid(np.arange(n_y * px_per_tile), np.arange(n_x * px_per_tile), indexing="ij")
    is_light = (ix // px_per_tile + iy // px_per_tile) % 2 == 0
    return np.where(is_light[..., None], 200, 110).astype(np.uint8).repeat(3, axis=-1)


class Go2RoughEnv:
    def __init__(
        self, num_envs, env_cfg, obs_cfg, reward_cfg, command_cfg, terrain_cfg, show_viewer=False, camera_res=None,
        rendered_envs_idx=(0,), camera_far=20.0,
    ):
        self.num_envs = num_envs
        self.num_actions = env_cfg["num_actions"]
        self.num_commands = command_cfg["num_commands"]
        self.device = gs.device
        self.cfg = env_cfg

        self.dt = 0.02
        self.max_episode_length = math.ceil(env_cfg["episode_length_s"] / self.dt)

        self.env_cfg = env_cfg
        self.obs_cfg = obs_cfg
        self.reward_cfg = reward_cfg
        self.command_cfg = command_cfg
        self.terrain_cfg = terrain_cfg
        self.obs_scales = obs_cfg["obs_scales"]
        self.reward_scales = reward_cfg["reward_scales"]

        self.scene = gs.Scene(
            sim_options=gs.options.SimOptions(dt=self.dt, substeps=2),
            rigid_options=gs.options.RigidOptions(
                enable_self_collision=False,
                max_collision_pairs=60,
                tolerance=1e-5,
                # モーターの速度上限を使うときは、環境ごとに違うトルク上限を設定できるようにする
                batch_dofs_info="dof_vel_limits" in env_cfg,
            ),
            # 描画する環境 (たくさんのロボットを 1 枚の画像に写す動画では複数を指定する)
            vis_options=gs.options.VisOptions(rendered_envs_idx=list(rendered_envs_idx)),
            viewer_options=gs.options.ViewerOptions(
                camera_pos=(2.0, 0.0, 2.5),
                camera_lookat=(0.0, 0.0, 0.5),
                camera_fov=40,
            ),
            show_viewer=show_viewer,
        )

        # --- terrain ---
        height_field = make_height_field(terrain_cfg)
        # 表示用: terrain_cfg["checker_size"] [m] を与えると地面を市松模様にする (物理には影響しない)
        terrain_surface = None
        if terrain_cfg.get("checker_size"):
            checker = make_checker_image(
                height_field.shape, terrain_cfg["horizontal_scale"], terrain_cfg["checker_size"]
            )
            terrain_surface = gs.surfaces.Default(diffuse_texture=gs.textures.ImageTexture(image_array=checker))
        self.scene.add_entity(
            gs.morphs.Terrain(
                height_field=height_field,
                horizontal_scale=terrain_cfg["horizontal_scale"],
                vertical_scale=terrain_cfg["vertical_scale"],
            ),
            surface=terrain_surface,
        )
        self.height_field = torch.tensor(
            height_field * terrain_cfg["vertical_scale"], dtype=gs.tc_float, device=gs.device
        )
        self.terrain_extent = torch.tensor(
            [(s - 1) * terrain_cfg["horizontal_scale"] for s in height_field.shape], dtype=gs.tc_float, device=gs.device
        )

        self.robot = self.scene.add_entity(
            gs.morphs.URDF(
                # urdf_file のない古い cfgs.pkl (Go2 Sprawling 導入前に学習したモデル) は Genesis 同梱の Go2 を使う
                file=env_cfg.get("urdf_file", "urdf/go2/urdf/go2.urdf"),
                pos=env_cfg["base_init_pos"],
                quat=env_cfg["base_init_quat"],
            ),
        )

        # 動画撮影用のオフスクリーンカメラ (scene.build より前に追加する必要がある)
        self.camera = None
        if camera_res is not None:
            self.camera = self.scene.add_camera(res=camera_res, fov=50, GUI=False, far=camera_far)

        self.scene.build(n_envs=num_envs)

        # --- joints / PD ---
        self.motors_dof_idx = torch.tensor(
            [self.robot.get_joint(name).dof_start for name in env_cfg["joint_names"]], dtype=gs.tc_int, device=gs.device
        )
        self.actions_dof_idx = torch.argsort(self.motors_dof_idx)
        self.robot.set_dofs_kp([env_cfg["kp"]] * self.num_actions, self.motors_dof_idx)
        self.robot.set_dofs_kv([env_cfg["kd"]] * self.num_actions, self.motors_dof_idx)
        # モーターの速度上限 (Genesis は URDF の velocity を使わない)。関節名に含まれる "hip" / "thigh" / "calf" ごとに指定する
        self.dof_vel_limits = None
        if "dof_vel_limits" in env_cfg:
            self.dof_vel_limits = torch.tensor(
                [next(v for k, v in env_cfg["dof_vel_limits"].items() if k in name) for name in env_cfg["joint_names"]],
                dtype=gs.tc_float,
                device=gs.device,
            )
            self.dof_force_limits = self.robot.get_dofs_force_range(self.motors_dof_idx)[1].reshape(-1, self.num_actions)[0]

        # --- links used for contacts (Go2 の足先は calf リンクに統合されている) ---
        link_names = [link.name for link in self.robot.links]
        self.base_link_idx = link_names.index("base")
        self.feet_link_idx = torch.tensor(
            [i for i, n in enumerate(link_names) if n.endswith("_calf")], dtype=gs.tc_int, device=gs.device
        )
        # 足先 (球の中心) の calf リンク座標での位置 (URDF の *_foot_joint)
        self.foot_offset = torch.tensor(env_cfg.get("foot_offset", [0.0, 0.0, -0.213]), dtype=gs.tc_float, device=gs.device)
        self.hip_dof_idx = [i for i, name in enumerate(env_cfg["joint_names"]) if "hip" in name]
        self.penalized_link_idx = torch.tensor(
            [i for i, n in enumerate(link_names) if n.endswith(("_thigh", "_hip"))], dtype=gs.tc_int, device=gs.device
        )

        # --- terrain curriculum ---
        n_levels = terrain_cfg["n_levels"]
        self.terrain_types = torch.arange(num_envs, device=gs.device) % len(TERRAIN_TYPES)
        self.terrain_levels = torch.randint(
            0, terrain_cfg["max_init_level"] + 1, (num_envs,), device=gs.device
        )
        self.max_terrain_level = n_levels - 1
        self.env_origins = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.expected_distance = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.is_curriculum_active = False

        # --- height scan points (胴体の yaw 座標系) ---
        xs = torch.tensor(obs_cfg["height_scan_x"], dtype=gs.tc_float, device=gs.device)
        ys = torch.tensor(obs_cfg["height_scan_y"], dtype=gs.tc_float, device=gs.device)
        grid_x, grid_y = torch.meshgrid(xs, ys, indexing="ij")
        self.scan_points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=-1)
        self.num_scan_points = self.scan_points.shape[0]

        # --- buffers ---
        self.global_gravity = torch.tensor([0.0, 0.0, -1.0], dtype=gs.tc_float, device=gs.device)
        self.init_base_quat = torch.tensor(env_cfg["base_init_quat"], dtype=gs.tc_float, device=gs.device)
        self.inv_base_init_quat = inv_quat(self.init_base_quat)
        self.default_dof_pos = torch.tensor(
            [env_cfg["default_joint_angles"][name] for name in env_cfg["joint_names"]], dtype=gs.tc_float, device=gs.device
        )
        self.init_dof_pos_qpos_order = torch.tensor(
            [env_cfg["default_joint_angles"][joint.name] for joint in self.robot.joints[1:]],
            dtype=gs.tc_float,
            device=gs.device,
        )

        self.base_pos = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_quat = torch.zeros((num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.base_euler = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_lin_vel = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.base_ang_vel = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.projected_gravity = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.measured_heights = torch.zeros((num_envs, self.num_scan_points), dtype=gs.tc_float, device=gs.device)
        self.dof_pos = torch.zeros((num_envs, self.num_actions), dtype=gs.tc_float, device=gs.device)
        self.dof_vel = torch.zeros_like(self.dof_pos)
        self.actions = torch.zeros_like(self.dof_pos)
        self.last_actions = torch.zeros_like(self.dof_pos)
        self.contact_forces = torch.zeros((num_envs, len(link_names), 3), dtype=gs.tc_float, device=gs.device)
        self.feet_air_time = torch.zeros((num_envs, 4), dtype=gs.tc_float, device=gs.device)
        self.last_feet_contact = torch.zeros((num_envs, 4), dtype=gs.tc_bool, device=gs.device)
        self.feet_pos = torch.zeros((num_envs, 4, 3), dtype=gs.tc_float, device=gs.device)
        self.feet_vel = torch.zeros_like(self.feet_pos)
        # ロボット全体 (全リンク) の重心の位置と速度 (速度は位置の差分)
        self.com_pos = torch.zeros((num_envs, 3), dtype=gs.tc_float, device=gs.device)
        self.com_vel = torch.zeros_like(self.com_pos)
        self.rew_buf = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.reset_buf = torch.ones((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.episode_length_buf = torch.zeros((num_envs,), dtype=gs.tc_int, device=gs.device)
        self.commands = torch.zeros((num_envs, self.num_commands), dtype=gs.tc_float, device=gs.device)
        self.commands_scale = torch.tensor(
            [self.obs_scales["lin_vel"], self.obs_scales["lin_vel"], self.obs_scales["ang_vel"]],
            dtype=gs.tc_float,
            device=gs.device,
        )
        self.commands_limits = tuple(
            torch.tensor(values, dtype=gs.tc_float, device=gs.device)
            for values in zip(
                command_cfg["lin_vel_x_range"], command_cfg["lin_vel_y_range"], command_cfg["ang_vel_range"]
            )
        )
        # 速度指令のカリキュラム: 前後の速度の範囲を初期範囲から始め、追従できたら最終範囲 (lin_vel_x_range) まで広げる
        self.lin_vel_x_limit = command_cfg.get("lin_vel_x_init_range", command_cfg["lin_vel_x_range"])[1]
        if command_cfg.get("curriculum", False):
            self.commands_limits[0][0], self.commands_limits[1][0] = -self.lin_vel_x_limit, self.lin_vel_x_limit
        self.is_command_curriculum_active = False
        # ジャンプ指令 (command_cfg["jump"] があるときだけ): 指令を受けると jump_cfg["window_s"] 秒の間、速度指令を 0 にして
        # その場で跳ぶ。跳んで着地するまでを「指令中」とし、観測には指令中かどうかと指令からの経過時間を加える
        self.jump_cfg = command_cfg.get("jump")
        self.num_jump_obs = 2 if self.jump_cfg else 0
        self.jump_timer = torch.full((num_envs,), -1.0, dtype=gs.tc_float, device=gs.device)  # -1: 指令なし
        self.is_jump_pending = torch.zeros((num_envs,), dtype=gs.tc_bool, device=gs.device)
        self.jump_flight_time = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        # 指令を受けた時点の、足元の地面からの重心の高さ
        self.jump_start_com_height = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)
        self.feet_contact = torch.zeros((num_envs, 4), dtype=gs.tc_bool, device=gs.device)
        # ジャンプ中は胴体の高さ・上下の速度・基準姿勢からのずれの罰を外す
        self.jump_masked_rewards = set(reward_cfg.get("jump_masked_rewards", []))
        self.command_curriculum_stats = torch.zeros(2, dtype=gs.tc_float, device=gs.device)
        self.common_step_counter = 0
        self.extras = {}

        self.reward_functions, self.episode_sums = {}, {}
        for name in self.reward_scales:
            self.reward_scales[name] *= self.dt
            self.reward_functions[name] = getattr(self, "_reward_" + name)
            self.episode_sums[name] = torch.zeros((num_envs,), dtype=gs.tc_float, device=gs.device)

        self.reset()
        self.is_curriculum_active = terrain_cfg["curriculum"]
        self.is_command_curriculum_active = command_cfg.get("curriculum", False)

    # ------------------------------------------------------------------ terrain helpers

    def _subterrain_centers(self, levels, types):
        size, border = self.terrain_cfg["subterrain_size"], self.terrain_cfg["border_size"]
        return torch.stack((border + (levels + 0.5) * size, border + (types + 0.5) * size), dim=-1).to(gs.tc_float)

    def _terrain_heights(self, points_xy):
        """ワールド座標 xy (..., 2) の地面の高さ [m] (最近傍セル)。"""
        idx = torch.round(points_xy / self.terrain_cfg["horizontal_scale"]).long()
        idx[..., 0].clamp_(0, self.height_field.shape[0] - 1)
        idx[..., 1].clamp_(0, self.height_field.shape[1] - 1)
        return self.height_field[idx[..., 0], idx[..., 1]]

    def _update_height_scan(self):
        yaw = torch.deg2rad(self.base_euler[:, 2])
        cos, sin = torch.cos(yaw)[:, None], torch.sin(yaw)[:, None]
        px, py = self.scan_points[None, :, 0], self.scan_points[None, :, 1]
        world_xy = torch.stack(
            (self.base_pos[:, None, 0] + cos * px - sin * py, self.base_pos[:, None, 1] + sin * px + cos * py), dim=-1
        )
        self.measured_heights = self._terrain_heights(world_xy)

    def _get_com_pos(self):
        return self.scene.sim.rigid_solver.get_links_root_COM(links_idx=[self.robot.base_link.idx])[:, 0]

    def _get_feet_pos(self):
        pos = self.robot.get_links_pos()[:, self.feet_link_idx]
        quat = self.robot.get_links_quat()[:, self.feet_link_idx]
        return pos + transform_by_quat(self.foot_offset.expand(pos.shape), quat)

    def _update_curriculum(self, env_ids):
        distance = torch.norm(self.base_pos[env_ids, :2] - self.env_origins[env_ids, :2], dim=1)
        move_up = distance > self.terrain_cfg["subterrain_size"] / 2
        move_down = (distance < self.expected_distance[env_ids] * 0.5) & ~move_up
        levels = self.terrain_levels[env_ids] + move_up.long() - move_down.long()
        # 最高難易度をクリアした環境はランダムな難易度に戻し、易しい地形を忘れないようにする
        levels = torch.where(
            levels > self.max_terrain_level, torch.randint_like(levels, 0, self.max_terrain_level + 1), levels
        )
        self.terrain_levels[env_ids] = levels.clamp(min=0)

    def _update_command_curriculum(self, env_ids):
        # 2 秒以上続いたエピソードの、1 ステップあたりの前後・横速度の追従報酬 (最大 1) の平均で判定する。
        # 速く走るほど地形の端に早く着いて短く終わるので、エピソードの長さで割って比べる
        # リセットのたびに集計し、指令の切り替え間隔 (resampling_time_s) ごとにまとめて判定する
        is_long = self.episode_length_buf[env_ids] * self.dt > 2.0
        sums = self.episode_sums["tracking_lin_vel"][env_ids][is_long]
        steps = self.episode_length_buf[env_ids][is_long]
        self.command_curriculum_stats += torch.stack(
            ((sums / (steps * self.reward_scales["tracking_lin_vel"])).sum(), is_long.sum().float())
        )

    def _expand_commands(self):
        tracking_sum, count = self.command_curriculum_stats.tolist()
        self.command_curriculum_stats.zero_()
        if count > 0 and tracking_sum / count > self.command_cfg["curriculum_threshold"]:
            final = self.command_cfg["lin_vel_x_range"][1]
            self.lin_vel_x_limit = min(self.lin_vel_x_limit + self.command_cfg["curriculum_step"], final)
            self.commands_limits[0][0], self.commands_limits[1][0] = -self.lin_vel_x_limit, self.lin_vel_x_limit

    def _apply_motor_speed_limits(self):
        # 回っている向きに出せるトルクを、速度上限の 70% から上限にかけて最大トルクから 0 まで直線的に下げる
        # (逆向き = ブレーキ側は最大トルクのまま)。実際のモーターの「速いほどトルクが出ない」特性の簡易版
        knee = 0.7
        ratio = ((self.dof_vel_limits - self.dof_vel.abs()) / ((1.0 - knee) * self.dof_vel_limits)).clamp(0.0, 1.0)
        limited = self.dof_force_limits * ratio
        upper = torch.where(self.dof_vel > 0, limited, self.dof_force_limits)
        lower = torch.where(self.dof_vel < 0, -limited, -self.dof_force_limits)
        self.robot.set_dofs_force_range(lower, upper, self.motors_dof_idx)

    def trigger_jump(self, env_ids):
        """ジャンプ指令を出す (すでに指令中の環境はそのまま)。"""
        env_ids = env_ids[self.jump_timer[env_ids] < 0]
        self.jump_timer[env_ids] = 0.0
        self.is_jump_pending[env_ids] = True
        self.jump_flight_time[env_ids] = 0.0
        self.jump_start_com_height[env_ids] = self._com_height()[env_ids]
        self.commands[env_ids] = 0.0

    def _trigger_random_jumps(self):
        # 平均 interval_s 秒に 1 回の割合でジャンプ指令を出す (エピソード開始直後と、学習の最初の start_iteration 回は除く)
        # rsl_rl は 1 iteration で num_steps_per_env (24) ステップ進める
        if self.common_step_counter < self.jump_cfg.get("start_iteration", 0) * 24:
            return
        is_ready = (self.jump_timer < 0) & (self.episode_length_buf * self.dt > 1.0)
        is_triggered = is_ready & (torch.rand(self.num_envs, device=gs.device) < self.dt / self.jump_cfg["interval_s"])
        self.trigger_jump(is_triggered.nonzero(as_tuple=False)[:, 0])

    def _update_jump(self):
        is_active = self.jump_timer >= 0
        self.jump_timer = torch.where(is_active, self.jump_timer + self.dt, self.jump_timer)
        # 4 本とも浮いている時間が min_flight_s 以上続いたあとに接地したら、ジャンプ完了 (指令中を解除)
        is_airborne = ~self.feet_contact.any(dim=1)
        has_landed = self.is_jump_pending & ~is_airborne & (self.jump_flight_time >= self.jump_cfg["min_flight_s"])
        self.jump_flight_time = torch.where(self.is_jump_pending & is_airborne, self.jump_flight_time + self.dt, 0.0)
        self.is_jump_pending &= ~has_landed
        # 時間切れ: 跳べなかった場合も含めて指令を解除し、速度指令を引き直す
        is_over = is_active & (self.jump_timer >= self.jump_cfg["window_s"])
        self.jump_timer[is_over] = -1.0
        self.is_jump_pending &= ~is_over
        self._resample_commands(is_over.nonzero(as_tuple=False)[:, 0])
        self._trigger_random_jumps()

    # ------------------------------------------------------------------ core

    def _resample_commands(self, env_ids):
        if len(env_ids) == 0:
            return
        commands = gs_rand(*self.commands_limits, (len(env_ids),))
        # 小さな並進指令は 0 にして、その場で止まる / 旋回する動作も学習させる
        is_small = torch.norm(commands[:, :2], dim=1) < self.command_cfg["min_lin_vel"]
        commands[is_small, :2] = 0.0
        self.commands[env_ids] = commands

    def step(self, actions):
        self.actions = torch.clip(actions, -self.env_cfg["clip_actions"], self.env_cfg["clip_actions"])
        exec_actions = self.last_actions if self.env_cfg["simulate_action_latency"] else self.actions
        target_dof_pos = exec_actions * self.env_cfg["action_scale"] + self.default_dof_pos
        if self.dof_vel_limits is not None:
            self._apply_motor_speed_limits()
        self.robot.control_dofs_position(target_dof_pos[:, self.actions_dof_idx], slice(6, 18))
        self.scene.step()

        self.episode_length_buf += 1
        self.common_step_counter += 1
        if self.is_command_curriculum_active and self.common_step_counter % int(self.env_cfg["resampling_time_s"] / self.dt) == 0:
            self._expand_commands()
        self.base_pos = self.robot.get_pos()
        self.base_quat = self.robot.get_quat()
        self.base_euler = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, self.base_quat), rpy=True, degrees=True
        )
        inv_base_quat = inv_quat(self.base_quat)
        self.base_lin_vel = transform_by_quat(self.robot.get_vel(), inv_base_quat)
        self.base_ang_vel = transform_by_quat(self.robot.get_ang(), inv_base_quat)
        self.projected_gravity = transform_by_quat(self.global_gravity, inv_base_quat)
        self.dof_pos = self.robot.get_dofs_position(self.motors_dof_idx)
        self.dof_vel = self.robot.get_dofs_velocity(self.motors_dof_idx)
        self.contact_forces = self.robot.get_links_net_contact_force()
        # 足先の速度は前ステップからの位置の差分で求める (足先は calf リンクの原点ではないため)
        feet_pos = self._get_feet_pos()
        self.feet_vel = (feet_pos - self.feet_pos) / self.dt
        self.feet_pos = feet_pos
        com_pos = self._get_com_pos()
        self.com_vel = (com_pos - self.com_pos) / self.dt
        self.com_pos = com_pos
        self._update_height_scan()
        self.expected_distance += torch.norm(self.commands[:, :2], dim=1) * self.dt

        # feet contact / air time
        feet_contact = torch.norm(self.contact_forces[:, self.feet_link_idx], dim=-1) > 1.0
        self.feet_contact = feet_contact
        if self.jump_cfg:
            self._update_jump()
        # 接触判定のちらつき対策として前ステップの接触とも OR を取る
        contact_filt = feet_contact | self.last_feet_contact
        self.first_contact = (self.feet_air_time > 0.0) & contact_filt
        self.feet_air_time += self.dt

        self.rew_buf.zero_()
        for name, reward_func in self.reward_functions.items():
            rew = reward_func() * self.reward_scales[name]
            if name in self.jump_masked_rewards:
                rew = rew * (self.jump_timer < 0)
            self.rew_buf += rew
            self.episode_sums[name] += rew
        if self.reward_cfg["only_positive_rewards"]:
            # 学習初期に罰ばかりで「早く転んで終わる」方策に落ちるのを防ぐ
            self.rew_buf.clamp_(min=0.0)

        self.feet_air_time *= ~contact_filt
        self.last_feet_contact = feet_contact

        resample_ids = (self.episode_length_buf % int(self.env_cfg["resampling_time_s"] / self.dt) == 0).nonzero(
            as_tuple=False
        )[:, 0]
        self._resample_commands(resample_ids)
        self.commands[self.jump_timer >= 0] = 0.0

        # termination
        is_timeout = self.episode_length_buf > self.max_episode_length
        is_out_of_bounds = (
            (self.base_pos[:, :2] < 1.0).any(dim=1) | (self.base_pos[:, :2] > self.terrain_extent - 1.0).any(dim=1)
        )
        is_base_contact = torch.norm(self.contact_forces[:, self.base_link_idx], dim=-1) > 1.0
        is_tilted = self.projected_gravity[:, 2] > -self.env_cfg["termination_min_gravity_z"]
        if self.jump_cfg and self.jump_cfg.get("is_flip", False):
            # 宙返りの指令中は逆さまになるので、傾きでは終了しない
            is_tilted &= self.jump_timer < 0
        # terminate_on_fall=False: 失敗しても時間切れまで続ける (罰の多い報酬で、早く終わらせて罰を避ける方策に落ちないように)
        self.reset_buf = (is_base_contact | is_tilted) & self.env_cfg.get("terminate_on_fall", True)
        self.reset_buf |= is_timeout | is_out_of_bounds
        self.reset_buf |= self.scene.get_error_envs_mask()
        # 時間切れ・地形の端への到達は失敗ではないので、価値関数のブートストラップ対象にする
        self.extras["time_outs"] = (is_timeout | is_out_of_bounds).to(dtype=gs.tc_float)

        self._reset_idx(self.reset_buf.nonzero(as_tuple=False)[:, 0])
        self._update_observation()

        self.last_actions.copy_(self.actions)
        return self.get_observations(), self.rew_buf, self.reset_buf, self.extras

    def _reset_idx(self, env_ids):
        if len(env_ids) == 0:
            return
        if self.is_curriculum_active:
            self._update_curriculum(env_ids)
        if self.is_command_curriculum_active:
            self._update_command_curriculum(env_ids)

        n = len(env_ids)
        origins_xy = self._subterrain_centers(self.terrain_levels[env_ids], self.terrain_types[env_ids])
        # マスの中心から ±spawn_spread [m] の範囲にばらまく (動画でたくさん並べるときは広げて重なりを減らす)
        spread = self.env_cfg.get("spawn_spread", 0.5)
        origins_xy += gs_rand(
            torch.tensor([-spread, -spread], device=gs.device), torch.tensor([spread, spread], device=gs.device), (n,)
        )
        # 胴体の下の範囲で一番高い地面に合わせて出発高さを決める
        footprint = torch.tensor(
            [[0.0, 0.0], [0.3, 0.2], [0.3, -0.2], [-0.3, 0.2], [-0.3, -0.2]], dtype=gs.tc_float, device=gs.device
        )
        ground = self._terrain_heights(origins_xy[:, None, :] + footprint[None]).max(dim=1).values
        self.env_origins[env_ids, :2] = origins_xy
        self.env_origins[env_ids, 2] = ground

        yaw = (torch.rand(n, device=gs.device) * 2.0 - 1.0) * math.pi
        quat = torch.stack(
            (torch.cos(yaw / 2), torch.zeros_like(yaw), torch.zeros_like(yaw), torch.sin(yaw / 2)), dim=-1
        )
        pos = torch.cat((origins_xy, (ground + self.env_cfg["base_init_pos"][2])[:, None]), dim=-1)
        qpos = torch.cat((pos, quat, self.init_dof_pos_qpos_order.expand(n, -1)), dim=-1)
        self.robot.set_qpos(qpos, envs_idx=env_ids, zero_velocity=True)

        self.base_pos[env_ids] = pos
        self.base_quat[env_ids] = quat
        self.base_euler[env_ids] = quat_to_xyz(
            transform_quat_by_quat(self.inv_base_init_quat, quat), rpy=True, degrees=True
        )
        self.projected_gravity[env_ids] = self.global_gravity
        self.base_lin_vel[env_ids] = 0.0
        self.base_ang_vel[env_ids] = 0.0
        self.dof_pos[env_ids] = self.default_dof_pos
        self.dof_vel[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.last_actions[env_ids] = 0.0
        self.feet_air_time[env_ids] = 0.0
        self.feet_pos[env_ids] = self._get_feet_pos()[env_ids]
        self.feet_vel[env_ids] = 0.0
        self.com_pos[env_ids] = self._get_com_pos()[env_ids]
        self.com_vel[env_ids] = 0.0
        self.jump_timer[env_ids] = -1.0
        self.is_jump_pending[env_ids] = False
        self.jump_flight_time[env_ids] = 0.0
        self.last_feet_contact[env_ids] = False
        self.episode_length_buf[env_ids] = 0
        self.expected_distance[env_ids] = 0.0

        self.extras["episode"] = {}
        for key, value in self.episode_sums.items():
            self.extras["episode"]["rew_" + key] = value[env_ids].mean() / self.env_cfg["episode_length_s"]
            value[env_ids] = 0.0
        self.extras["episode"]["terrain_level"] = self.terrain_levels.float().mean()
        self.extras["episode"]["max_command_x"] = self.lin_vel_x_limit

        self._resample_commands(env_ids)

    def _update_observation(self):
        self._update_height_scan()
        heights = torch.clip(
            self.base_pos[:, 2:3] - self.reward_cfg["base_height_target"] - self.measured_heights, -1.0, 1.0
        )
        self.obs_buf = torch.cat(
            (
                self.base_ang_vel * self.obs_scales["ang_vel"],  # 3
                self.projected_gravity,  # 3
                self.commands * self.commands_scale,  # 3
                self._jump_observation(),  # num_jump_obs (0 or 2)
                (self.dof_pos - self.default_dof_pos) * self.obs_scales["dof_pos"],  # 12
                self.dof_vel * self.obs_scales["dof_vel"],  # 12
                self.actions,  # 12
                heights * self.obs_scales["height_measurements"],  # num_scan_points
            ),
            dim=-1,
        )
        self.privileged_obs_buf = self.base_lin_vel * self.obs_scales["lin_vel"]

    def _jump_observation(self):
        if not self.jump_cfg:
            return torch.zeros((self.num_envs, 0), dtype=gs.tc_float, device=gs.device)
        phase = torch.where(self.jump_timer >= 0, self.jump_timer / self.jump_cfg["window_s"], 0.0)
        return torch.stack((self.is_jump_pending.float(), phase), dim=-1)

    def get_observations(self):
        return TensorDict(
            {"policy": self.obs_buf, "privileged": self.privileged_obs_buf}, batch_size=[self.num_envs]
        )

    def reset(self):
        self.reset_buf.fill_(True)
        self._reset_idx(torch.arange(self.num_envs, device=gs.device))
        self._update_observation()
        return self.get_observations()

    # ------------------------------------------------------------------ rewards

    def _reward_tracking_lin_vel(self):
        lin_vel_error = torch.sum(torch.square(self.commands[:, :2] - self.base_lin_vel[:, :2]), dim=1)
        return torch.exp(-lin_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_tracking_ang_vel(self):
        ang_vel_error = torch.square(self.commands[:, 2] - self.base_ang_vel[:, 2])
        return torch.exp(-ang_vel_error / self.reward_cfg["tracking_sigma"])

    def _reward_lin_vel_z(self):
        return torch.square(self.base_lin_vel[:, 2])

    def _reward_ang_vel_xy(self):
        return torch.sum(torch.square(self.base_ang_vel[:, :2]), dim=1)

    def _reward_base_height(self):
        # 足元の地面 (高さスキャンの平均) からの高さ
        height = self.base_pos[:, 2] - self.measured_heights.mean(dim=1)
        return torch.square(height - self.reward_cfg["base_height_target"])

    def _reward_action_rate(self):
        return torch.sum(torch.square(self.last_actions - self.actions), dim=1)

    def _reward_similar_to_default(self):
        return torch.sum(torch.abs(self.dof_pos - self.default_dof_pos), dim=1)

    def _reward_feet_air_time(self):
        # 着地した瞬間に、それまでの滞空時間が閾値より長ければ報酬 (短い小刻みなステップは減点)
        rew = torch.sum((self.feet_air_time - self.reward_cfg["feet_air_time_target"]) * self.first_contact, dim=1)
        return rew * (torch.norm(self.commands[:, :2], dim=1) > 0.1)

    def _reward_feet_slip(self):
        # 接地している足先の水平方向の滑り
        feet_contact = torch.norm(self.contact_forces[:, self.feet_link_idx], dim=-1) > 1.0
        return torch.sum(torch.norm(self.feet_vel[..., :2], dim=-1) * feet_contact, dim=1)

    def _reward_hip_deviation(self):
        # 付け根 (鉛直軸まわり) の基準姿勢からの振れ。大きく弧を描く振り出しを抑える
        return torch.sum(torch.square(self.dof_pos[:, self.hip_dof_idx] - self.default_dof_pos[self.hip_dof_idx]), dim=1)

    def _com_height(self):
        return self.com_pos[:, 2] - self.measured_heights.mean(dim=1)

    def jump_com_apex(self):
        """重心が放物線で達する最高点の、指令を受けた時点からの上昇量 [m]。

        空中では重心は脚の動かし方によらず放物線を描くので、浮いている間はこの値が一定になる。
        脚を縮めて足先を持ち上げても増えず、体ごと跳び上がったときだけ増える。
        """
        rise_to_apex = torch.clip(self.com_vel[:, 2], min=0.0) ** 2 / (2.0 * 9.81)
        return self._com_height() + rise_to_apex - self.jump_start_com_height

    def _jump_height(self):
        # 足元の地面からの胴体の高さの、基準の高さからの上昇量
        height = self.base_pos[:, 2] - self.measured_heights.mean(dim=1) - self.reward_cfg["base_height_target"]
        return torch.clip(height, 0.0, self.jump_cfg["max_height"])

    def _reward_jump_stretch(self):
        # ジャンプ指令中 (着地まで) の胴体の上昇量。脚を伸ばして体を持ち上げる動きを見つけやすくする小さな報酬
        return self._jump_height() * self.is_jump_pending

    def _reward_jump_com(self):
        # ジャンプ指令中に 4 本とも浮いている間の、重心の最高点の上昇量 (jump_com_apex)
        is_airborne = ~self.feet_contact.any(dim=1)
        apex = torch.clip(self.jump_com_apex(), 0.0, self.jump_cfg["max_com_rise"])
        return apex * (self.is_jump_pending & is_airborne)

    def _reward_jump_flight_pose(self):
        # 空中での基準姿勢からのずれ。足を胴体に引きつける (跳ね上げる) 動きを抑え、着地に備えて脚を下ろさせる
        is_airborne = ~self.feet_contact.any(dim=1)
        deviation = torch.sum(torch.square(self.dof_pos - self.default_dof_pos), dim=1)
        return deviation * (self.is_jump_pending & is_airborne)

    def _reward_collision(self):
        # 太もも・股関節リンクの接触 (段差のふちへのぶつかり)
        return torch.sum(torch.norm(self.contact_forces[:, self.penalized_link_idx], dim=-1) > 0.1, dim=1).float()
