from __future__ import annotations
import torch
import numpy as np
from typing import TYPE_CHECKING, Tuple

from isaaclab.managers import SceneEntityCfg
import isaacsim.core.utils.torch as torch_utils
from isaacsim.core.utils.torch import quat_apply, tf_combine, get_euler_xyz
# We import FactoryTask to type-hint the config object we expect to receive
from rsdr_isaaclab.tasks.manager_based.factory.factory_tasks_cfg import FactoryTask

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
from ..factory_utils import (
    get_held_base_pose, 
    get_target_held_base_pose, 
    get_keypoint_offsets, 
    squashing_fn,
    wrap_yaw
)
# -------------------------------------------------------------------------
# Helper Functions
# -------------------------------------------------------------------------

def _wrap_yaw(angle: torch.Tensor) -> torch.Tensor:
    """Helper: Ensure yaw stays within range [-2pi, 2pi]."""
    return torch.where(angle > np.deg2rad(235), angle - 2 * np.pi, angle)

# -------------------------------------------------------------------------
# Reward Terms
# -------------------------------------------------------------------------

def factory_keypoint_optimization(
    env: ManagerBasedRLEnv,
    held_asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
    factory_task_cfg: any,
    keypoint_level: str,
) -> torch.Tensor:
    """Reward for aligning virtual keypoints using identical DirectRLEnv logic."""
    
    # 1. Get Assets
    held_asset = env.scene[held_asset_cfg.name]
    fixed_asset = env.scene[fixed_asset_cfg.name]

    # 2. Get World Poses relative to env_origins (Identical to FactoryEnv._compute_intermediate_values)
    held_pos = held_asset.data.root_pos_w - env.scene.env_origins
    held_quat = held_asset.data.root_quat_w
    fixed_pos = fixed_asset.data.root_pos_w - env.scene.env_origins
    fixed_quat = fixed_asset.data.root_quat_w

    # 3. Use IDENTICAL functions from factory_utils
    held_base_pos, held_base_quat = get_held_base_pose(
        held_pos, held_quat, factory_task_cfg.name, factory_task_cfg.fixed_asset_cfg, env.num_envs, env.device
    )
    target_held_base_pos, target_held_base_quat = get_target_held_base_pose(
        fixed_pos, fixed_quat, factory_task_cfg.name, factory_task_cfg.fixed_asset_cfg, env.num_envs, env.device
    )

    # 4. Compute Keypoints (Identical loop to factory_env.py _get_factory_rew_dict)
    num_kp = factory_task_cfg.num_keypoints
    keypoints_held = torch.zeros((env.num_envs, num_kp, 3), device=env.device)
    keypoints_fixed = torch.zeros((env.num_envs, num_kp, 3), device=env.device)
    
    offsets = get_keypoint_offsets(num_kp, env.device)
    keypoint_offsets = offsets * factory_task_cfg.keypoint_scale
    
    for idx, keypoint_offset in enumerate(keypoint_offsets):
        # Replicates the tf_combine logic exactly
        _, keypoints_held[:, idx] = torch_utils.tf_combine(
            held_base_quat, held_base_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
            keypoint_offset.repeat(env.num_envs, 1),
        )
        _, keypoints_fixed[:, idx] = torch_utils.tf_combine(
            target_held_base_quat, target_held_base_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1),
            keypoint_offset.repeat(env.num_envs, 1),
        )

    # 5. Final Distance & Squash
    keypoint_dist = torch.norm(keypoints_held - keypoints_fixed, p=2, dim=-1).mean(-1)
    
    # Pull coefs based on level
    if keypoint_level == "baseline":
        a, b = factory_task_cfg.keypoint_coef_baseline
    elif keypoint_level == "coarse":
        a, b = factory_task_cfg.keypoint_coef_coarse
    else:
        a, b = factory_task_cfg.keypoint_coef_fine
        
    return squashing_fn(keypoint_dist, a, b)

def factory_pose_alignment_reward(
    env: ManagerBasedRLEnv,
    held_asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
    factory_task_cfg: any,
    mode: str,
) -> torch.Tensor:
    """Binary reward using identical DirectRLEnv success logic."""
    
    # 1. Get Base Poses
    held_pos = env.scene[held_asset_cfg.name].data.root_pos_w - env.scene.env_origins
    held_quat = env.scene[held_asset_cfg.name].data.root_quat_w
    fixed_pos = env.scene[fixed_asset_cfg.name].data.root_pos_w - env.scene.env_origins
    fixed_quat = env.scene[fixed_asset_cfg.name].data.root_quat_w

    held_base_pos, _ = get_held_base_pose(
        held_pos, held_quat, factory_task_cfg.name, factory_task_cfg.fixed_asset_cfg, env.num_envs, env.device
    )
    target_held_base_pos, _ = get_target_held_base_pose(
        fixed_pos, fixed_quat, factory_task_cfg.name, factory_task_cfg.fixed_asset_cfg, env.num_envs, env.device
    )

    # 2. Alignment Logic (Matches DirectRLEnv _get_curr_successes)
    xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
    z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
    
    is_centered = xy_dist < 0.0025 # 2.5mm threshold
    
    # 3. Height Threshold
    fixed_cfg = factory_task_cfg.fixed_asset_cfg
    success_threshold = factory_task_cfg.success_threshold if mode == "success" else factory_task_cfg.engage_threshold
    
    if factory_task_cfg.name in ["peg_insert", "gear_mesh"]:
        height_threshold = fixed_cfg.height * success_threshold
    else: # nut_thread
        height_threshold = fixed_cfg.thread_pitch * success_threshold
        
    is_close_or_below = z_disp < height_threshold
    
    result = torch.logical_and(is_centered, is_close_or_below)

    # 4. Rotation Check (Only for Nut Thread Success)
    if mode == "success" and factory_task_cfg.name == "nut_thread":
        # Get current fingertip yaw (DirectRLEnv uses fingertip_midpoint_quat for rot check)
        # Re-resolve the robot fingertip if needed, or use held_quat if the nut is fixed in hand
        fingertip_quat = env.scene["robot"].data.body_quat_w[:, env.scene["robot"].body_names.index("panda_fingertip_centered")]
        
        _, _, curr_yaw = torch_utils.get_euler_xyz(fingertip_quat)
        curr_yaw = wrap_yaw(curr_yaw) #
        
        # Check if yaw is below the success threshold defined in config
        is_rotated = curr_yaw < factory_task_cfg.ee_success_yaw
        result = torch.logical_and(result, is_rotated)

    return result.float()


def factory_action_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    arm_action = env.action_manager.get_term("arm_action")
    return torch.norm(arm_action.actions, p=2, dim=-1)


def factory_action_rate_l2_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    arm_action = env.action_manager.get_term("arm_action")
    return torch.norm(arm_action.action - arm_action.prev_action, p=2, dim=-1)