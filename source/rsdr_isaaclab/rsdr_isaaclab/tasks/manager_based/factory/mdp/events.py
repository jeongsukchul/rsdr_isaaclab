# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING


from isaaclab.managers import SceneEntityCfg
from isaacsim.core.utils.torch import quat_from_euler_xyz, tf_combine, tf_inverse
from rsdr_isaaclab.tasks.manager_based.factory import factory_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
# --- Main Event Function ---
def randomize_task_space_gains(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    action_term_name: str,
    stiffness_distribution_params: tuple[float, float],
    damping_distribution_params: tuple[float, float],
):
    """
    Randomizes the Kp and Kd gains of the FactoryTaskSpaceControl action term.
    Distribution params are (min, max) scaling factors (e.g., 0.8 to 1.2).
    """
    # 1. Get the Action Term instance
    # We use the method that definitely works now
    action_term = env.action_manager.action_terms[action_term_name]

    # 2. Randomize Stiffness (Kp)
    # Get the nominal (blueprint) value from config
    nominal_kp = torch.tensor(action_term.cfg.task_prop_gains, device=env.device)
    
    # Generate random scales (Uniform distribution)
    low, high = stiffness_distribution_params
    scales_kp = torch.rand((len(env_ids), 6), device=env.device) * (high - low) + low
    
    # Update the LIVE tensor
    action_term.kp[env_ids] = nominal_kp * scales_kp

    # 3. Randomize Damping (Kd)
    nominal_kd = torch.tensor(action_term.cfg.task_deriv_gains, device=env.device)
    low_d, high_d = damping_distribution_params
    scales_kd = torch.rand((len(env_ids), 6), device=env.device) * (high_d - low_d) + low_d
    
    action_term.kd[env_ids] = nominal_kd * scales_kd


def set_body_inertias(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    asset_cfg: SceneEntityCfg,
    offset_value: float = 0.01,
):
    """
    Increases the diagonal inertia of the robot links to improve simulation stability.
    """
    asset = env.scene[asset_cfg.name]
    
    # Get all inertias
    inertias = asset.root_physx_view.get_inertias()
    
    # If env_ids is None, apply to all environments
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    env_ids = env_ids.to(inertias.device)
    # Create offset for specified environments
    offset = torch.zeros_like(inertias[env_ids])
    offset[:, :, [0, 4, 8]] += offset_value
    
    # Apply offset
    new_inertias = inertias[env_ids] + offset
    asset.root_physx_view.set_inertias(new_inertias, env_ids)



from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import FRAME_MARKER_CFG

def update_debug_vis(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None = None):
    # 1. Lazy Initialization
    arm_action = env.action_manager.get_term("arm_action")
    if not hasattr(env, "debug_ee_marker"):
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.prim_path = "/Visuals/ee_frame"
        
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05) 
        
        env.debug_ee_marker = VisualizationMarkers(marker_cfg)

    # 2. Get Data
    robot = env.scene["robot"]
    idx = robot.body_names.index("panda_fingertip_centered")

    ee_pos = robot.data.body_pos_w[:, idx]
    ee_quat = robot.data.body_quat_w[:, idx]
    
    # 3. Update
    env.debug_ee_marker.visualize(translations=ee_pos, orientations=ee_quat)


