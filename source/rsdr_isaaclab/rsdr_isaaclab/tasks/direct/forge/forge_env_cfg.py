# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_env_cfg import OBS_DIM_CFG, STATE_DIM_CFG, CtrlCfg, FactoryEnvCfg, ObsRandCfg

from .forge_events import randomize_dead_zone
from .forge_tasks_cfg import ForgeGearMesh, ForgeNutThread, ForgePegInsert, ForgeTask
from rsdr_isaaclab.tasks.direct.samplers.gbs_sampler import GBS

OBS_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})

STATE_DIM_CFG.update({"force_threshold": 1, "ft_force": 3})


@configclass
class ForgeCtrlCfg(CtrlCfg):
    ema_factor_range = [0.025, 0.1]
    default_task_prop_gains = [565.0, 565.0, 565.0, 28.0, 28.0, 28.0]
    task_prop_gains_noise_level = [0.41, 0.41, 0.41, 0.41, 0.41, 0.41]
    pos_threshold_noise_level = [0.25, 0.25, 0.25]
    rot_threshold_noise_level = [0.29, 0.29, 0.29]
    default_dead_zone = [5.0, 5.0, 5.0, 1.0, 1.0, 1.0]


@configclass
class ForgeObsRandCfg(ObsRandCfg):
    fingertip_pos = 0.00025
    fingertip_rot_deg = 0.1
    ft_force = 1.0


@configclass
class EventCfg:
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "mass_distribution_params": (-0.005, 0.005),
            "operation": "add",
            "distribution": "uniform",
        },
    )

    held_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("held_asset"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    fixed_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("fixed_asset"),
            "static_friction_range": (0.25, 1.25),  # TODO: Set these values based on asset type.
            "dynamic_friction_range": (0.25, 0.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 128,
        },
    )

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.75, 0.75),
            "dynamic_friction_range": (0.75, 0.75),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 1,
        },
    )

    dead_zone_thresholds = EventTerm(
        func=randomize_dead_zone, mode="interval", interval_range_s=(2.0, 2.0)  # (0.25, 0.25)
    )


@configclass
class ForgeEnvCfg(FactoryEnvCfg):
    action_space: int = 7
    dr_update_batch_size: int = 1024
    obs_rand: ForgeObsRandCfg = ForgeObsRandCfg()
    ctrl: ForgeCtrlCfg = ForgeCtrlCfg()
    task: ForgeTask = ForgeTask()
    events: EventCfg = EventCfg()

    ft_smoothing_factor: float = 0.25

    obs_order: list = [
        "fingertip_pos_rel_fixed",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "ft_force",
        "force_threshold",
    ]
    state_order: list = [
        "fingertip_pos",
        "fingertip_quat",
        "ee_linvel",
        "ee_angvel",
        "joint_pos",
        "held_pos",
        "held_pos_rel_fixed",
        "held_quat",
        "fixed_pos",
        "fixed_quat",
        "task_prop_gains",
        "ema_factor",
        "ft_force",
        "pos_threshold",
        "rot_threshold",
        "force_threshold",
    ]


@configclass
class ForgeTaskPegInsertCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    task = ForgePegInsert()
    episode_length_s = 10.0


@configclass
class ForgeTaskPegInsert_GBS_Cfg(ForgeTaskPegInsertCfg):
    sampler_class = GBS
    sampler_kwargs = dict(
        beta=1.0,
        batch_size=1024,
        init_std=0.5,
        lr=1e-4,
        clip_grad=1.0,
        num_steps=100,
        model_num_layers=2,
        model_num_hid=64,
        max_rnd=1e8,
        sde_ctrl_noise=None,
        sde_ctrl_dropout=None,
        use_tanh_bijection=True,
        clip_prior_to_bounds=False,
        process_type="vp",
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        sigma_const=1.0,
        terminal_t=1.0,
        train_steps_per_update=1,
    )

    def __post_init__(self):
        self.sampler_kwargs = dict(self.sampler_kwargs)
        self.sampler_kwargs["batch_size"] = int(self.dr_update_batch_size)


@configclass
class ForgeTaskGearMeshCfg(ForgeEnvCfg):
    task_name = "gear_mesh"
    task = ForgeGearMesh()
    episode_length_s = 20.0


@configclass
class ForgeTaskGearMesh_GBS_Cfg(ForgeTaskGearMeshCfg):
    sampler_class = GBS
    sampler_kwargs = dict(
        beta=1.0,
        batch_size=1024,
        init_std=0.5,
        lr=1e-4,
        clip_grad=1.0,
        num_steps=100,
        model_num_layers=2,
        model_num_hid=64,
        max_rnd=1e8,
        sde_ctrl_noise=None,
        sde_ctrl_dropout=None,
        use_tanh_bijection=True,
        clip_prior_to_bounds=False,
        process_type="vp",
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        sigma_const=1.0,
        terminal_t=1.0,
        train_steps_per_update=1,
    )

    def __post_init__(self):
        self.sampler_kwargs = dict(self.sampler_kwargs)
        self.sampler_kwargs["batch_size"] = int(self.dr_update_batch_size)


@configclass
class ForgeTaskNutThreadCfg(ForgeEnvCfg):
    task_name = "nut_thread"
    task = ForgeNutThread()
    episode_length_s = 30.0


@configclass
class ForgeTaskNutThread_GBS_Cfg(ForgeTaskNutThreadCfg):
    sampler_class = GBS
    sampler_kwargs = dict(
        beta=1.0,
        batch_size=1024,
        init_std=0.5,
        lr=1e-4,
        clip_grad=1.0,
        num_steps=100,
        model_num_layers=2,
        model_num_hid=64,
        max_rnd=1e8,
        sde_ctrl_noise=None,
        sde_ctrl_dropout=None,
        use_tanh_bijection=True,
        clip_prior_to_bounds=False,
        process_type="vp",
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        sigma_const=1.0,
        terminal_t=1.0,
        train_steps_per_update=1,
    )

    def __post_init__(self):
        self.sampler_kwargs = dict(self.sampler_kwargs)
        self.sampler_kwargs["batch_size"] = int(self.dr_update_batch_size)
