# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.managers import SceneEntityCfg, ObservationTermCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import axis_angle_from_quat, quat_mul, quat_conjugate

if TYPE_CHECKING:
    pass

# =============================================================================
# Custom Factory Observations
# =============================================================================
def body_pos_rel_fixed(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg,
    body_name: SceneEntityCfg,
    action_term_name: str = "arm_action"
) -> torch.Tensor:
    """
    Computes the position of the robot fingertip relative to the NOISY fixed asset frame.
    Matches factory_env.py line 245 logic.
    """
    # 1. Get Entities
    robot = env.scene[robot_cfg.name]
    
    # 2. Get the Action Term to access the initialized noisy frame
    action_term = env.action_manager.get_term(action_term_name)
    
    # Matching DirectRLEnv: noisy_fixed_pos = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
    noisy_fixed_pos = action_term.fixed_pos_obs_frame + action_term.init_fixed_pos_obs_noise
    # 3. Get Fingertip Position relative to world origin (env-frame)
    # robot.data.body_pos_w is world-absolute; we need environment-relative
    fingertip_body_idx = robot.body_names.index(body_name)
    fingertip_midpoint_pos = robot.data.body_pos_w[:, fingertip_body_idx] - env.scene.env_origins
    output = fingertip_midpoint_pos - noisy_fixed_pos
    # 4. Compute Relative Position
    return output

def action_term_param(
    env: ManagerBasedRLEnv,
    action_term_name: str,
    param_name: str
) -> torch.Tensor:
    """
    Retrieves a LIVE parameter from the Action Term instance.
    """
    # 1. Get the Term Instance
    # Note: action_manager.action_terms is the dictionary mapping name -> term object
    action_term = env.action_manager.get_term(action_term_name)

    # 2. Map 'task_prop_gains' request to the internal 'kp' tensor
    # The config calls it 'task_prop_gains', but the class instance calls it 'kp'

    
    
    # 3. Access the Live Attribute  
    if hasattr(action_term, param_name):
        val = getattr(action_term, param_name)
    else:
        raise ValueError(f"Action term has no live attribute '{param_name}'")

    # 4. Return Tensor
    # If val is already a tensor [num_envs, N], return it
    if isinstance(val, torch.Tensor):
        return val
    
    # Fallback for static config values (like thresholds)
    return torch.tensor(val, device=env.device).repeat(env.num_envs, 1)

def FiniteDifferenceVelocityObs(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    action_term_name: str = "arm_action",
    data_type: str = "linear"
) -> torch.Tensor:
    """
    Retrieves the finite-differenced velocity stored in the Action Term.
    Matches factory_env.py lines 246-247.
    """
    action_term = env.action_manager.get_term(action_term_name)

    if data_type == "linear":
        # Returns ee_linvel_fd
        return action_term.ee_linvel_fd
    elif data_type == "angular":
        # Returns ee_angvel_fd
        return action_term.ee_angvel_fd
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

# =============================================================================
# Standard Physics Observations (Re-implemented for compatibility)
# =============================================================================

def body_world_position(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset body position in world frame."""
    asset = env.scene[asset_cfg.name]
    # Note: asset_cfg.body_ids is a list of indices resolved from body_names
    return asset.data.body_pos_w[:, asset_cfg.body_ids[0]]

def body_world_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset body orientation (quaternion) in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_quat_w[:, asset_cfg.body_ids[0]]

def body_world_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset body linear velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0]]

def body_world_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Asset body angular velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.body_ang_vel_w[:, asset_cfg.body_ids[0]]

def joint_pos(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Joint positions of the asset."""
    asset = env.scene[asset_cfg.name]
    # Note: asset_cfg.joint_ids is a list of indices resolved from joint_names
    return asset.data.joint_pos[:, asset_cfg.joint_ids]

def root_pos_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root position in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_pos_w

def root_quat_w(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Root orientation (quaternion) in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_quat_w

def last_action(env: ManagerBasedRLEnv, action_name: str | None = None) -> torch.Tensor:
    """The last input action to the environment."""
    if action_name is None:
        return env.action_manager.action
    else:
        return env.action_manager.get_term(action_name).raw_actions
    
def visualize_frames(env: ManagerBasedRLEnv):
    # 1. Get current world positions
    robot = env.scene["robot"]
    # World position of the centered fingertip frame
    ee_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_fingertip_centered")]
    
    # 2. Update the marker positions in the GUI
    # Note: env.scene["ee_marker"] must exist in your SceneCfg
    env.scene["ee_marker"].visualize(translations=ee_pos)
    
    # Observations must return a tensor, so return a dummy or the pos itself
    return ee_pos - env.scene.env_origins

