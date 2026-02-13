
from __future__ import annotations
import torch
from isaaclab.sim import SimulationContext
import numpy as np
from rsdr_isaaclab.tasks.manager_based.factory import factory_utils
import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
from typing import TYPE_CHECKING
from isaaclab.managers import SceneEntityCfg
import carb
from .. import factory_control

from isaaclab.actuators import ImplicitActuator


def _extract_param(sampler, master_values, param_name, device):
    """Finds the indices for 'param_name' and returns the values."""
    for p in sampler.cfg.params:
        if p.name == param_name:
            # master_values is (num_envs, total_params)
            return master_values[:, p.indices]
    
    # Fallback if param not in config (Safety)
    return torch.zeros((master_values.shape[0], 3), device=device) # Assume size 3 default

def apply_learned_randomization(
    env, 
    env_ids, 
    robot_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
    task_cfg: any,
):
    """
    Master Reset Function.
    1. Samples Learned Dynamics & States (The "Brain").
    2. Applies Physics DR (Mass, Stiffness, Gravity).
    3. Calls IK Reset with the Sampled Poses.
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    num_envs = len(env_ids)
    sampler = env.sampler
    if not hasattr(env, "fixed_pos_obs_frame"):
        env.fixed_pos_obs_frame = torch.zeros((env.num_envs, 3), device=env.device)
    if not hasattr(env, "init_fixed_pos_obs_noise"):
        env.init_fixed_pos_obs_noise = torch.zeros((env.num_envs, 3), device=env.device)
    # --- 1. Master Sampling ---
    # Sample the Big Tensor [num_envs, total_params]
    master_values = sampler.sample(num_envs)
    log_probs = sampler.log_prob(master_values)
    # Store log_probs for the RL/Sampler Update
    if "dr_samples" not in env.extras:
        env.extras["dr_samples"] = torch.zeros((env.num_envs, sampler.num_params), device=env.device)
        env.extras["dr_log_probs"] = torch.zeros((env.num_envs,), device=env.device)
    
    # Save values for the resetting environments
    env.extras["dr_samples"][env_ids] = master_values
    env.extras["dr_log_probs"][env_ids] = log_probs
    stiff_val = None
    damping_val = None
    # --- 2. Physics Dispatch (Apply before kinematics) ---
    for p_cfg in sampler.cfg.params:
        vals = master_values[:, p_cfg.indices]
        
        if p_cfg.event_type == "stiffness":
            stiff_val = vals
        elif p_cfg.event_type == "damping":
            damping_val =vals
        elif p_cfg.event_type == "mass":
            randomize_mass(env, env_ids, p_cfg.target_asset, vals, p_cfg.target_indices)
        elif p_cfg.event_type == "gravity":
            randomize_gravity(env, vals)
    randomize_actuator_gain(env, env_ids, "robot", stiff_val, damping_val)
    # --- 3. State/IK Dispatch ---
    # Pass the sampled values to your existing IK function
    reset_factory_assets_with_ik(
        env, env_ids, robot_cfg, held_asset_cfg, fixed_asset_cfg, 
        task_cfg, master_values=master_values
    )

# rsdr_isaaclab/tasks/manager_based/factory/mdp/randomization.py

import torch
from isaaclab.actuators import ImplicitActuator

def randomize_actuator_gain(env, env_ids, asset_name, stiff_values, damping_values):
    asset = env.scene[asset_name]
    asset_cfg = SceneEntityCfg("robot")
    joint_ids = asset_cfg.joint_ids

    for actuator in asset.actuators.values():
        if isinstance(joint_ids, slice):
            # we take all the joints of the actuator
            actuator_indices = slice(None)
            if isinstance(actuator.joint_indices, slice):
                global_indices = slice(None)
            elif isinstance(actuator.joint_indices, torch.Tensor):
                global_indices = actuator.joint_indices.to(asset.device)
            else:
                raise TypeError("Actuator joint indices must be a slice or a torch.Tensor.")
        elif isinstance(actuator.joint_indices, slice):
            # we take the joints defined in the asset config
            global_indices = actuator_indices = torch.tensor(joint_ids, device=asset.device)
        else:
            # we take the intersection of the actuator joints and the asset config joints
            actuator_joint_indices = actuator.joint_indices
            asset_joint_ids = torch.tensor(joint_ids, device=asset.device)
            # the indices of the joints in the actuator that have to be randomized
            actuator_indices = torch.nonzero(torch.isin(actuator_joint_indices, asset_joint_ids)).view(-1)
            if len(actuator_indices) == 0:
                continue
            # maps actuator indices that have to be randomized to global joint indices
            global_indices = actuator_joint_indices[actuator_indices]
            stiffness = actuator.stiffness[env_ids].clone()
            stiffness[:, actuator_indices] = asset.data.default_joint_stiffness[env_ids][
                    :, global_indices
                ].clone() *  stiff_values
            actuator.stiffness[env_ids] = stiffness 
            asset.write_joint_stiffness_to_sim(
                stiffness, joint_ids=actuator.joint_indices, env_ids=env_ids
            )
            damping = actuator.damping[env_ids].clone()
            damping[:, actuator_indices] = asset.data.default_joint_damping[env_ids][:, global_indices].clone() *damping_values
            actuator.damping[env_ids] = damping
            if isinstance(actuator, ImplicitActuator):
                asset.write_joint_damping_to_sim(damping, joint_ids=actuator.joint_indices, env_ids=env_ids)

def randomize_mass(env, env_ids, asset_name, values, body_ids=None):
    asset = env.scene[asset_name]
    if body_ids is None: body_ids = torch.arange(asset.num_bodies, device=env.device)
    
    # Get masses for specific bodies
    masses = asset.root_physx_view.get_masses().clone()
    
    # Apply randomization: default_mass[env, body] * value
    # values might be (num_envs, 1) or (num_envs, num_body_ids)
    default = asset.data.default_mass[env_ids[:, None], body_ids]
    
    # Update only the specific indices
    masses[env_ids[:, None], body_ids] = default * values
    asset.root_physx_view.set_masses(masses, env_ids)

def randomize_gravity(env, values):
    # values shape: (num_envs, 1)
    # Physics scene only supports 1 gravity vector globally
    g_z = values.mean().item()
    SimulationContext.instance().physics_sim_view.set_gravity(carb.Float3(0.0, 0.0, g_z))
    
def _set_franka_to_default_pose(env, env_ids, robot, task_cfg):
    # gripper open width (same as Direct)
    gripper_width = task_cfg.held_asset_cfg.diameter / 2 * 1.25

    joint_pos = robot.data.default_joint_pos[env_ids].clone()
    joint_pos[:, 7:9] = gripper_width

    reset_joints = torch.tensor(
        [0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0],
        device=env.device,
        dtype=joint_pos.dtype,
    )
    joint_pos[:, :7] = reset_joints[None, :]

    joint_vel = torch.zeros_like(joint_pos)
    joint_effort = torch.zeros_like(joint_pos)

    # IMPORTANT: set targets + write state (both)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

    # Sync internal buffers (Direct calls reset())
    # Depending on IsaacLab version, reset() may accept env_ids or not:
    try:
        robot.reset(env_ids=env_ids)
    except TypeError:
        robot.reset()

    robot.set_joint_effort_target(joint_effort, env_ids=env_ids)

    # Ensure sim/tensors reflect the new state (Direct steps once)
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)
def reset_factory_assets_with_ik(
    env,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
    task_cfg: any, 
    master_values: any = None,
):
    """Identical to FactoryEnv.randomize_initial_state."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)
    
    num_envs = len(env_ids)
    # ---- Direct buffers (allocate once) ----
    if not hasattr(env, "fixed_pos_obs_frame"):
        env.fixed_pos_obs_frame = torch.zeros((env.num_envs, 3), device=env.device)
    if not hasattr(env, "init_fixed_pos_obs_noise"):
        env.init_fixed_pos_obs_noise = torch.zeros((env.num_envs, 3), device=env.device)

    robot = env.scene[robot_cfg.name]
    held_asset = env.scene[held_asset_cfg.name]
    fixed_asset = env.scene[fixed_asset_cfg.name]
    sampler = env.sampler
    pos_idx = next(p.indices for p in sampler.cfg.params if p.name == "hand_init_pos_noise")
    orn_idx = next(p.indices for p in sampler.cfg.params if p.name == "hand_init_orn_noise")
    # We sample ONCE for all envs.
    noise_fixed_pos = _extract_param(sampler, master_values, "fixed_pos_noise", env.device)
    noise_fixed_yaw = _extract_param(sampler, master_values, "fixed_yaw_noise", env.device)
    noise_held_pos = _extract_param(sampler, master_values, "held_pos_noise", env.device)

    _set_franka_to_default_pose(env, env_ids, robot, task_cfg)
    env.scene.write_data_to_sim()
    robot.update(dt=0.0) 
    fixed_asset.update(dt=0.0)
    # 1. Randomize fixed asset pose
    fixed_state = fixed_asset.data.default_root_state[env_ids].clone()
    # rand_sample = torch.rand((num_envs, 3), device=env.device)
    # fixed_pos_noise = 2 * (rand_sample - 0.5) 
    fixed_state[:, 0:3] += noise_fixed_pos #(fixed_pos_noise @ torch.diag(torch.tensor(task_cfg.fixed_asset_init_pos_noise, device=env.device))) 
    fixed_state[:, 0:3] += env.scene.env_origins[env_ids] #
    
    # (1.b) Orientation Noise (Yaw only)
    fixed_orn_init_yaw = np.deg2rad(task_cfg.fixed_asset_init_orn_deg)
    # fixed_orn_yaw_range = np.deg2rad(task_cfg.fixed_asset_init_orn_range_deg)
    
    # rand_yaw = torch.rand((num_envs, 1), device=env.device)
    fixed_yaw = fixed_orn_init_yaw + noise_fixed_yaw.squeeze(-1) #fixed_orn_yaw_range * (2 * rand_yaw - 1.0).squeeze(-1) # [-1, 1] logic
    
    # Convert Yaw to Quat
    fixed_orn_quat = math_utils.quat_from_euler_xyz(
        torch.zeros_like(fixed_yaw), torch.zeros_like(fixed_yaw), fixed_yaw
    )
    fixed_state[:, 3:7] = fixed_orn_quat
    fixed_state[:, 7:] = 0.0 # Zero velocity
    
    # Write Fixed Asset
    fixed_asset.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
    fixed_asset.write_root_velocity_to_sim(fixed_state[:, 7:], env_ids=env_ids)
    fixed_asset.reset()
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    fixed_asset.update(dt=0.0)
    fixed_asset_pos_noise = torch.randn((num_envs, 3), dtype=torch.float32, device=env.device)
    fixed_asset_pos_rand = torch.tensor(env.cfg.obs_rand.fixed_asset_pos, dtype=torch.float32, device=env.device)
    fixed_asset_pos_noise = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)
    env.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise

    # ---- Direct-identical: compute fixed_pos_obs_frame (TIP FRAME) in env-frame ----
    fixed_pos_e = fixed_asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    fixed_quat = fixed_asset.data.root_quat_w[env_ids]

    fixed_tip_pos_local = torch.zeros((num_envs, 3), device=env.device)
    fixed_tip_pos_local[:, 2] += task_cfg.fixed_asset_cfg.height
    fixed_tip_pos_local[:, 2] += task_cfg.fixed_asset_cfg.base_height
    if task_cfg.name == "gear_mesh":
        fixed_tip_pos_local[:, 0] = task_cfg.fixed_asset_cfg.medium_gear_base_offset[0]

    _, fixed_tip_pos_e = torch_utils.tf_combine(
        fixed_quat,
        fixed_pos_e,
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).unsqueeze(0).repeat(num_envs, 1),
        fixed_tip_pos_local,
    )

    env.fixed_pos_obs_frame[env_ids] = fixed_tip_pos_e
    
    # 2. Get IK Target above Fixed Asset
    fixed_asset.update(dt=0.0)
    # fixed_pos = fixed_asset.data.root_pos_w[env_ids] - env.scene.env_origins[env_ids]
    # fixed_quat = fixed_asset.data.root_quat_w[env_ids]
    
    # fixed_tip_pos_local = torch.zeros((num_envs, 3), device=env.device)
    # fixed_tip_pos_local[:, 2] += task_cfg.fixed_asset_cfg.height + task_cfg.fixed_asset_cfg.base_height

    fixed_tip_pos = env.fixed_pos_obs_frame
    # 3. Iterative IK Loop
    bad_envs = env_ids.clone()
    ik_attempt = 0
    hand_init_pos_noise =  torch.tensor(task_cfg.hand_init_pos_noise, device=env.device)
    hand_init_orn_noise =  torch.tensor(task_cfg.hand_init_orn_noise, device=env.device)
    
    default_joints = torch.tensor([0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], device=env.device)
    while True:
        if len(bad_envs) == 0 or ik_attempt > 5: # Break after 5 attempts
            break
        n_bad = len(bad_envs)

        #reset to default joints
        new_samples = sampler.sample(len(bad_envs))
        robot.data.joint_pos[bad_envs, 0:7] = default_joints
        robot.write_joint_state_to_sim(robot.data.joint_pos[bad_envs], torch.zeros_like(robot.data.joint_pos[bad_envs]), env_ids=bad_envs)
        robot.update(dt=0.0)

        new_samples = sampler.sample(len(bad_envs))
        current_pos_noise = _extract_param(sampler, new_samples, "hand_init_pos_noise", env.device)
        current_orn_noise = _extract_param(sampler, new_samples, "hand_init_orn_noise", env.device)
        env.extras["dr_samples"][bad_envs[:, None], pos_idx] = current_pos_noise
        env.extras["dr_samples"][bad_envs[:, None], orn_idx] = current_orn_noise
        updated_rows = env.extras["dr_samples"][bad_envs]
        env.extras["dr_log_probs"][bad_envs] = sampler.log_prob(updated_rows)
        # current_pos_noise = 2 * (torch.rand((n_bad, 3), device=env.device) - 0.5) @ torch.diag(hand_init_pos_noise)
        # current_orn_noise = 2 * (torch.rand((n_bad, 3), device=env.device) - 0.5) @ torch.diag(hand_init_orn_noise)
        # Compute targets for bad envs
        target_pos = fixed_tip_pos[bad_envs] + torch.tensor(task_cfg.hand_init_pos, device=env.device)
        target_pos += current_pos_noise
        base_euler = torch.tensor(task_cfg.hand_init_orn, device=env.device).repeat(n_bad, 1)
        base_euler += current_orn_noise
        
        target_quat = torch_utils.quat_from_euler_xyz(
            base_euler[:, 0], base_euler[:, 1], base_euler[:, 2]
        )
        fingertip_idx = robot.body_names.index("panda_fingertip_centered")

        # Iterative IK refinement loop
        for _ in range(30):
            # Compute error using EXACT DirectRLEnv function
            curr_pos = robot.data.body_pos_w[bad_envs, fingertip_idx] - env.scene.env_origins[bad_envs]
            curr_quat = robot.data.body_quat_w[bad_envs, fingertip_idx]

            pos_error, axis_angle_error = factory_control.get_pose_error(
                fingertip_midpoint_pos=curr_pos,
                fingertip_midpoint_quat=curr_quat,
                ctrl_target_fingertip_midpoint_pos=target_pos,
                ctrl_target_fingertip_midpoint_quat=target_quat,
                jacobian_type="geometric",
                rot_error_type="axis_angle",
            )

            jac = robot.root_physx_view.get_jacobians()[bad_envs, robot.body_names.index("panda_leftfinger")-1, 0:6, 0:7] * 0.5+ \
                          robot.root_physx_view.get_jacobians()[bad_envs, robot.body_names.index("panda_rightfinger")-1, 0:6, 0:7] * 0.5
            # Solve DLS IK using EXACT DirectRLEnv function
            delta_dof_pos = factory_control.get_delta_dof_pos(
                delta_pose=torch.cat((pos_error, axis_angle_error), dim=-1),
                ik_method="dls",
                jacobian=jac,
                device=env.device,
            )
            robot.data.joint_pos[bad_envs, 0:7] += delta_dof_pos[:, 0:7]
            robot.write_joint_state_to_sim(robot.data.joint_pos[bad_envs], torch.zeros_like(robot.data.joint_pos[bad_envs]), env_ids=bad_envs)

        robot.update(dt=0.0)
        # Check for convergence
        final_pos = robot.data.body_pos_w[bad_envs, fingertip_idx] - env.scene.env_origins[bad_envs]
        final_quat = robot.data.body_quat_w[bad_envs, fingertip_idx]
        pe, ae = factory_control.get_pose_error(final_pos, final_quat, target_pos, target_quat, "geometric", "axis_angle")
        success = (torch.norm(pe, dim=1) < 5e-3) & (torch.norm(ae, dim=1) < 5e-3)
        still_bad = ~success
        bad_envs = bad_envs[still_bad]
        # Reset bad ones to default before next try
        if len(bad_envs) > 0:
            # rand_pos = 2 * (torch.rand((len(bad_envs), 3), device=env.device) - 0.5) @ torch.diag(hand_init_pos_noise)
            # target_pos[bad_envs] = env.fixed_pos_obs_frame[bad_envs] + torch.tensor(task_cfg.hand_init_pos, device=env.device) + rand_pos
            # Use specific reset joints
            # default_joints = torch.tensor([0.00871, -0.10368, -0.00794, -1.49139, -0.00083, 1.38774, 0.0], device=env.device)
            # robot.data.joint_pos[bad_envs, 0:7] = default_joints
            # robot.write_joint_state_to_sim(robot.data.joint_pos[bad_envs], torch.zeros_like(robot.data.joint_pos[bad_envs]), env_ids=bad_envs)
            _set_franka_to_default_pose(env, bad_envs, robot, task_cfg) #
        ik_attempt += 1
    #Update robot kinematics chain after IK
    robot.update(dt=0.0)

    # --- 3. Teleport Held Asset (Using Corrected Transform) ---
    # Get current (post-IK) hand pose
    fingertip_idx = robot.body_names.index("panda_fingertip_centered")
    fingertip_pos = robot.data.body_pos_w[env_ids, fingertip_idx]
    fingertip_quat = robot.data.body_quat_w[env_ids, fingertip_idx]

    # 4. Teleport Held Asset with Z-Flip
    flip_z_quat = torch.tensor([0.0, 0.0, 1.0, 0.0], device=env.device).repeat(num_envs, 1)
    # Combine current hand pose with Z-flip
    fingertip_flipped_quat, fingertip_flipped_pos = torch_utils.tf_combine(
        fingertip_quat, fingertip_pos, flip_z_quat, torch.zeros_like(fingertip_pos)
    )
    # Apply Randomization to the Asset IN THE HAND
    held_asset_relative_pos, held_asset_relative_quat = factory_utils.get_handheld_asset_relative_pose(
        task_cfg=task_cfg,
        num_envs=num_envs, 
        device=env.device
    )

    held_asset_relative_pos = held_asset_relative_pos.to(dtype=torch.float32)
    held_asset_relative_quat = held_asset_relative_quat.to(dtype=torch.float32)
    # Ensure the inputs to tf_inverse are float32
    asset_in_hand_quat, asset_in_hand_pos = torch_utils.tf_inverse(
            held_asset_relative_quat, held_asset_relative_pos
        )
    # Calculate final world pose for the asset
    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        fingertip_flipped_quat, fingertip_flipped_pos, asset_in_hand_quat, asset_in_hand_pos
    )
    # Apply Randomization to the Asset IN THE HAND
    # rand_sample = torch.rand((num_envs, 3), device=env.device)
    # held_noise = 2 * (rand_sample - 0.5) # [-1, 1]

    # Specific logic for Gear Mesh
    if task_cfg.name == "gear_mesh":
        noise_held_pos[:, 2] = -torch.abs(noise_held_pos[:, 2]) #torch.rand((num_envs,), device=env.device) # [-1, 0]
    # noise_scale = torch.tensor(task_cfg.held_asset_pos_noise, device=env.device)
    # held_noise = held_noise @torch.diag(noise_scale)

    #inject noise in current frame
    translated_held_asset_quat, translated_held_asset_pos = torch_utils.tf_combine(
        translated_held_asset_quat, translated_held_asset_pos, 
        torch.tensor([1.0, 0.0, 0.0, 0.0], device=env.device).repeat(num_envs, 1),
        noise_held_pos,
    )

    #Write Held Asset
    held_state = held_asset.data.default_root_state[env_ids].clone()
    held_state[:, 0:3] = translated_held_asset_pos # Already includes env_origins via fingertip_pos
    held_state[:, 3:7] = translated_held_asset_quat
    held_state[:, 7:] = 0.0
    held_asset.write_root_pose_to_sim(held_state[:, 0:7], env_ids=env_ids)
    held_asset.write_root_velocity_to_sim(held_state[:, 7:], env_ids=env_ids)
    held_asset.reset()
    # Establish Grasp (Final joint write)
    obj_width = factory_utils.get_asset_grasp_width(task_cfg)
    # Target = Width/2 - epsilon (for friction squeeze)
    grasp_joint_pos = max(0.0, (obj_width / 2.0) - 0.001) 
    
    # Teleport fingers to SURFACE
    robot.data.joint_pos[env_ids, 7:9] = grasp_joint_pos
    robot.write_joint_state_to_sim(robot.data.joint_pos[env_ids], torch.zeros_like(robot.data.joint_pos[env_ids]), env_ids=env_ids)
    robot.set_joint_position_target(robot.data.joint_pos, env_ids=env_ids)
    if task_cfg.name == "gear_mesh" and getattr(task_cfg, "add_flanking_gears", False):
        small_gear = env.scene["small_gear"]
        large_gear = env.scene["large_gear"]
        
        # Teleport to same location as fixed asset (bases overlap)
        small_gear.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        small_gear.write_root_velocity_to_sim(torch.zeros_like(fixed_state[:, 7:]), env_ids=env_ids)
        small_gear.reset()
        large_gear.write_root_pose_to_sim(fixed_state[:, 0:7], env_ids=env_ids)
        large_gear.write_root_velocity_to_sim(torch.zeros_like(fixed_state[:, 7:]), env_ids=env_ids)
        large_gear.reset()


    # (A) get the SAME action term used during rollouts
    arm_term = env.action_manager.get_term("arm_action")

    # (B) during reset, use the reset gains (like DirectEnv)
    #     (these fields must exist on your action term if you use action_term_param obs)
    reset_task_prop_gains = torch.tensor(
        env.cfg.actions.arm_action.reset_task_prop_gains, device=env.device, dtype=torch.float32
    ).unsqueeze(0).repeat(env.num_envs, 1)

    arm_term.task_prop_gains[:] = reset_task_prop_gains
    arm_term.task_deriv_gains[:] = factory_utils.get_deriv_gains(
        reset_task_prop_gains, env.cfg.actions.arm_action.reset_rot_deriv_scale
    )

    # (C) zero action => "hold current pose" (same as DirectEnv close_gripper_in_place)
    zero_action = torch.zeros((env.num_envs, 6), device=env.device)
    # if your term uses EMA internally, calling this each step is the safest
    grasp_steps = int(0.25 / env.physics_dt)

    # for _ in range(grasp_steps):
        # 1) close gripper using physx PD (same idea as DirectEnv)
    joint_pos_tgt = robot.data.joint_pos.clone()
    joint_pos_tgt[env_ids, 7:9] = grasp_joint_pos #0.0
    robot.set_joint_position_target(joint_pos_tgt, env_ids=env_ids)

    # 2) torque-control the arm via the action term
    # arm_term.process_actions(zero_action)
    # arm_term.apply_actions()

    # 3) step physics
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt=env.physics_dt)

    # (D) restore default gains for the episode (like DirectEnv)
    # default_task_prop_gains = torch.tensor(
    #     env.cfg.actions.arm_action.default_task_prop_gains, device=env.device, dtype=torch.float32
    # ).unsqueeze(0).repeat(env.num_envs, 1)

    # arm_term.task_prop_gains[:] = default_task_prop_gains
    # arm_term.task_deriv_gains[:] = factory_utils.get_deriv_gains(default_task_prop_gains)