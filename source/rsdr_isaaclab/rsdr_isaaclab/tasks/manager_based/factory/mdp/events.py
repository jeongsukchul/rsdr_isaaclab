# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import numpy as np
from typing import TYPE_CHECKING

import isaacsim.core.utils.torch as torch_utils

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

import torch
import numpy as np
import isaacsim.core.utils.torch as torch_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.envs import ManagerBasedRLEnv

# Import the shared factory control module used by the DirectRLEnv
from .. import factory_control

def _set_franka_to_default_pose(env, env_ids, robot, task_cfg):
    """Return Franka to its default joint position."""
    gripper_width = task_cfg.held_asset_cfg.diameter / 2 * 1.25 #
    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, 7:] = gripper_width # MIMIC joints
    
    # Use reset joints from the task config
    reset_joints = torch.tensor([0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], device=env.device)
    joint_pos[:, :7] = reset_joints[None, :]
    
    robot.write_joint_state_to_sim(joint_pos, torch.zeros_like(joint_pos), env_ids=env_ids)
    # Note: Manager-based environments use write_joint_state_to_sim to teleport


def reset_factory_assets_with_ik(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
    task_cfg: any, 
):
    """Identical to FactoryEnv.randomize_initial_state."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs = len(env_ids)
    robot = env.scene[robot_cfg.name]
    held_asset = env.scene[held_asset_cfg.name]
    fixed_asset = env.scene[fixed_asset_cfg.name]

    _set_franka_to_default_pose(env, env_ids, robot, task_cfg)
    env.scene.write_data_to_sim()
    robot.update(dt=0.0) 
    fixed_asset.update(dt=0.0)
    # 1. Randomize fixed asset pose
    fixed_state = fixed_asset.data.default_root_state[env_ids].clone()
    rand_sample = torch.rand((num_envs, 3), device=env.device)
    fixed_pos_noise = 2 * (rand_sample - 0.5) 
    fixed_state[:, 0:3] += (fixed_pos_noise @ torch.diag(torch.tensor(task_cfg.fixed_asset_init_pos_noise, device=env.device))) 
    fixed_state[:, 0:3] += env.scene.env_origins[env_ids] #
    
    fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)

    
    
    # 2. Get IK Target above Fixed Asset
    fixed_asset.update(dt=0.0)
    fixed_pos = fixed_asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    fixed_quat = fixed_asset.data.root_quat_w[env_ids]
    
    fixed_tip_pos_local = torch.zeros((num_envs, 3), device=env.device)
    fixed_tip_pos_local[:, 2] += task_cfg.fixed_asset_cfg.height + task_cfg.fixed_asset_cfg.base_height
    
    _, fixed_tip_pos = torch_utils.tf_combine(
        fixed_quat, fixed_pos, 
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1), 
        fixed_tip_pos_local
    )


    # 3. Iterative IK Loop
    bad_envs = env_ids.clone()
    ik_attempt = 0
    while True:
        n_bad = len(bad_envs)
        # Compute targets for bad envs
        target_pos = fixed_tip_pos[bad_envs] + torch.tensor(task_cfg.hand_init_pos, device=env.device)
        target_quat = torch_utils.quat_from_euler_xyz(
            torch.tensor(task_cfg.hand_init_orn[0], device=env.device).repeat(n_bad),
            torch.tensor(task_cfg.hand_init_orn[1], device=env.device).repeat(n_bad),
            torch.tensor(task_cfg.hand_init_orn[2], device=env.device).repeat(n_bad)
        )

        # Iterative IK refinement loop
        for _ in range(30):
            # Compute error using EXACT DirectRLEnv function
            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=robot.data.body_pos_w[bad_envs, robot.body_names.index("panda_fingertip_centered")] - env.scene.env_origins[bad_envs],
                fingertip_midpoint_quat=robot.data.body_quat_w[bad_envs, robot.body_names.index("panda_fingertip_centered")],
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            # Solve DLS IK using EXACT DirectRLEnv function
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=torch.cat((pos_error, axis_angle_error), dim=-1),
                ik_method="dls",
                jacobian=(robot.root_physx_view.get_jacobians()[bad_envs, robot.body_names.index("panda_leftfinger")-1, 0:6, 0:7] + 
                          robot.root_physx_view.get_jacobians()[bad_envs, robot.body_names.index("panda_rightfinger")-1, 0:6, 0:7]) * 0.5,
                device=env.device,
            )
            robot.data.joint_pos[bad_envs, 0:7] += delta_dof_pos[:, 0:7]
            robot.write_joint_state_to_sim(robot.data.joint_pos[bad_envs], torch.zeros_like(robot.data.joint_pos[bad_envs]), env_ids=bad_envs)

        # Check for convergence
        any_error = torch.logical_or(torch.norm(pos_error, dim=1) > 1e-3, torch.norm(axis_angle_error, dim=1) > 1e-3)
        bad_envs = bad_envs[any_error.nonzero(as_tuple=False).squeeze(-1)]

        if len(bad_envs) == 0 or ik_attempt > 2: break
        _set_franka_to_default_pose(env, bad_envs, robot, task_cfg) #
        ik_attempt += 1

    # 4. Teleport Held Asset with Z-Flip
    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.device).repeat(num_envs, 1)
    fingertip_quat = robot.data.body_quat_w[env_ids, robot.body_names.index("panda_fingertip_centered")]
    fingertip_pos = robot.data.body_pos_w[env_ids, robot.body_names.index("panda_fingertip_centered")]

    # Combine current hand pose with Z-flip
    fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
        fingertip_quat, fingertip_pos, flip_z_quat, torch.zeros_like(fingertip_pos)
    )
    held_asset_relative_pos, held_asset_relative_quat = factory_utils.get_handheld_asset_relative_pose(
        task_cfg=task_cfg,
        num_envs=num_envs, 
        device=env.device
    )
    # Ensure the inputs to tf_inverse are float32
    held_asset_relative_pos = held_asset_relative_pos.to(dtype=torch.float32)
    held_asset_relative_quat = held_asset_relative_quat.to(dtype=torch.float32)

    # Now tf_inverse will work without the dtype RuntimeError
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
        held_asset_relative_quat, held_asset_relative_pos
    )
    # Calculate final world pose for the asset
    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        fingertip_flipped_quat, fingertip_flipped_pos, asset_in_hand_quat, asset_in_hand_pos
    )

    held_state = held_asset.data.default_root_state[env_ids].clone()
    held_state[:, 0:3] = translated_held_asset_pos # Already includes env_origins via fingertip_pos
    held_state[:, 3:7] = translated_held_asset_quat
    held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
    
    held_asset.update(dt=0.0)
    robot.update(dt=0.0) 

    # 5. Establish Grasp (Final joint write)
    robot.data.joint_pos[env_ids, 7:9] = 0.0 
    robot.write_joint_state_to_sim(robot.data.joint_pos[env_ids], torch.zeros_like(robot.data.joint_pos[env_ids]), env_ids=env_ids)
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
