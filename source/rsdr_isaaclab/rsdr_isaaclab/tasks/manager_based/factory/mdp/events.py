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


import torch
import isaacsim.core.utils.torch as torch_utils
import isaaclab.sim as sim_utils
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
def viz_fixed_obs_and_action_frames(
    env,
    env_ids: torch.Tensor,
    fixed_asset_cfg,
    task_cfg,
    noise_attr: str = "init_fixed_pos_obs_noise",   # must match what you actually use
):
    """Draw fixed_pos_obs_frame and fixed_pos_action_frame (=obs_frame + stored noise) in the viewport.

    IMPORTANT:
    - Positions in ManagerBased are usually world-frame already (root_pos_w).
    - If your stored noise is in env-frame (like DirectRLEnv), adding it to world positions is still correct
      because env-frame differs by translation only.
    """

    # ---- lazy-init markers (once) ----
    if not hasattr(env, "_viz_fixed_frames"):
        cfg = VisualizationMarkersCfg(
            prim_path="/World/Visuals/fixed_frames_debug",
            markers={
                # green frame: obs frame
                "obs": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
                ),
                # red frame: action frame (obs + noise)
                "action": sim_utils.UsdFileCfg(
                    usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/frame_prim.usd",
                    scale=(0.05, 0.05, 0.05),
                    visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                ),
            },
        )
        env._viz_fixed_frames = VisualizationMarkers(cfg)

    # ---- pick a small subset (otherwise 2*N markers every frame is heavy) ----
    if env_ids.numel() == 0:
        return

    fixed_asset = env.scene[fixed_asset_cfg.name]
    fixed_pos_w = fixed_asset.data.root_pos_w[env_ids]     # world frame
    fixed_quat_w = fixed_asset.data.root_quat_w[env_ids]

    # ---- compute fixed_pos_obs_frame (DirectRLEnv logic: bolt “tip” frame) ----
    fixed_tip_pos_local = torch.zeros((env_ids.numel(), 3), device=env.device)
    fixed_tip_pos_local[:, 2] = task_cfg.fixed_asset_cfg.height + task_cfg.fixed_asset_cfg.base_height
    if task_cfg.name == "gear_mesh":
        fixed_tip_pos_local[:, 0] = task_cfg.fixed_asset_cfg.medium_gear_base_offset[0]

    identity_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(env_ids.numel(), 1)
    obs_quat_w, obs_pos_w = torch_utils.tf_combine(
        q1=fixed_quat_w, t1=fixed_pos_w,
        q2=identity_quat, t2=fixed_tip_pos_local,
    )

    # ---- action frame = obs frame + stored noise ----
    noise = env.action_manager.get_term("arm_action").__getattribute__(noise_attr)
     # noise is in env-frame (DirectRLEnv logic), so we can add directly to world-frame positions
    if noise is None:
        print("couldn't get the noise data!")
        noise_sel = torch.zeros_like(obs_pos_w)
    else:
        noise_sel = noise[env_ids].to(obs_pos_w.dtype)

    action_pos_w = obs_pos_w + noise_sel
    action_quat_w = obs_quat_w

    # ---- pack into marker arrays (2 markers per env) ----
    translations = torch.cat([obs_pos_w, action_pos_w], dim=0).detach().cpu()
    orientations = torch.cat([obs_quat_w, action_quat_w], dim=0).detach().cpu()
    marker_indices = torch.tensor(
        [0] * env_ids.numel() + [1] * env_ids.numel(),
        dtype=torch.int64,
    )

    env._viz_fixed_frames.visualize(
        translations=translations,
        orientations=orientations,
        marker_indices=marker_indices,
    )