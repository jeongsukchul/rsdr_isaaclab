# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

import isaaclab.utils.math as math_utils
import isaacsim.core.utils.torch as torch_utils
from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# Import the factory control module
from .. import factory_control

# =============================================================================
# 1. Action Implementation
# =============================================================================

class FactoryTaskSpaceControl(ActionTerm):
    """
    Custom Task-Space Impedance Controller that replicates FactoryEnv logic exactly.
    """
    cfg: FactoryTaskSpaceControlCfg
    _asset: Articulation
    _fixed_asset: Articulation
    _held_asset: Articulation

    def __init__(self, cfg: FactoryTaskSpaceControlCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        

        # 1. Resolve Robot
        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, _ = self._asset.find_joints(cfg.joint_names)
        self._num_joints = len(self._joint_ids)
        
        # Find body indices (matching DirectRLEnv naming)
        self.left_finger_body_idx = self._asset.body_names.index("panda_leftfinger")
        self.right_finger_body_idx = self._asset.body_names.index("panda_rightfinger")
        self.fingertip_body_idx = self._asset.body_names.index("panda_fingertip_centered")

        # 2. Resolve Fixed and Held Assets
        self._fixed_asset = env.scene[cfg.fixed_asset_name]
        self._held_asset = env.scene[cfg.held_asset_name]

        # 3. Initialize Buffers
        self._raw_actions = torch.zeros(self.num_envs, 6, device=self.device)
        self._processed_actions = torch.zeros(self.num_envs, 6, device=self.device)
        
        # EMA smoothing (matching DirectRLEnv cfg.ema_factor)
        self.ema_factor = cfg.ema_factor
        self.actions = torch.zeros(self.num_envs, 6, device=self.device)
        self.prev_actions = torch.zeros(self.num_envs, 6, device=self.device)
        
        # Control targets (matching DirectRLEnv)
        self.ctrl_target_fingertip_midpoint_pos = torch.zeros(self.num_envs, 3, device=self.device)
        self.ctrl_target_fingertip_midpoint_quat = torch.zeros(self.num_envs, 4, device=self.device)
        self.ctrl_target_joint_pos = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)
        
        # Gains (public attributes for randomization/observation) - from cfg
        self.task_prop_gains = torch.tensor(
            cfg.default_task_prop_gains, device=self.device
        ).repeat(self.num_envs, 1)
        self.task_deriv_gains = self._get_deriv_gains(self.task_prop_gains)
        
        # Action thresholds (matching DirectRLEnv cfg)
        self.pos_threshold = torch.tensor(
            cfg.pos_action_threshold, device=self.device
        ).repeat(self.num_envs, 1)
        self.rot_threshold = torch.tensor(
            cfg.rot_action_threshold, device=self.device
        ).repeat(self.num_envs, 1)
        
        # Action bounds (matching DirectRLEnv cfg)
        self.pos_action_bounds = cfg.pos_action_bounds
        self.rot_action_bounds = cfg.rot_action_bounds
        
        # Observation Noise Buffer (persists across steps, resets on reset)
        self.init_fixed_pos_obs_noise = torch.zeros(self.num_envs, 3, device=self.device)
        self.fixed_pos_obs_frame = torch.zeros(self.num_envs, 3, device=self.device)
        
        # Dead zone thresholds (optional, for force unreliability simulation)
        self.dead_zone_thresholds = None
        
        # Finite-differencing tensors (matching DirectRLEnv)
        self.last_update_timestamp = 0.0
        self.prev_fingertip_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.prev_fingertip_quat = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).unsqueeze(0).repeat(self.num_envs, 1)
        self.prev_joint_pos = torch.zeros((self.num_envs, 7), device=self.device)
        
        # Computed intermediate values (matching DirectRLEnv)
        self.fingertip_midpoint_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.fingertip_midpoint_linvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_angvel = torch.zeros((self.num_envs, 3), device=self.device)
        self.fingertip_midpoint_jacobian = torch.zeros((self.num_envs, 6, 7), device=self.device)
        self.arm_mass_matrix = torch.zeros((self.num_envs, 7, 7), device=self.device)
        self.joint_pos = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)
        self.joint_vel = torch.zeros((self.num_envs, self._asset.num_joints), device=self.device)
        
        # Finite-differenced velocities (more reliable)
        self.ee_linvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.ee_angvel_fd = torch.zeros((self.num_envs, 3), device=self.device)
        self.joint_vel_fd = torch.zeros((self.num_envs, 7), device=self.device)
        
        # Fixed and held asset states
        self.fixed_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.fixed_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self.held_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self.held_quat = torch.zeros((self.num_envs, 4), device=self.device)
        
        # Store task config for unidirectional_rot check
        self.unidirectional_rot = env.cfg.task.unidirectional_rot if hasattr(env.cfg.task, 'unidirectional_rot') else False

        class ConfigWrapper:
            class Scene:
                num_envs = self.num_envs
            scene = Scene()
            ctrl = self.cfg  # Our action config has all ctrl parameters

        
        self._ctrl_cfg = ConfigWrapper()
    def _get_deriv_gains(self, prop_gains, rot_deriv_scale=1.0):
        """Compute derivative gains from proportional gains (matching DirectRLEnv factory_utils)."""
        deriv_gains = 2 * torch.sqrt(prop_gains)
        deriv_gains[:, 3:] *= rot_deriv_scale
        return deriv_gains

    @property
    def action_dim(self) -> int:
        return 6

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        
        # 1. Reset internal smoothing/EMA buffers
        self._raw_actions[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        
        # 2. Re-initialize observation noise for the fixed asset
        fixed_asset_pos_noise = torch.randn((len(env_ids), 3), device=self.device)
        fixed_asset_pos_rand = torch.tensor(self.cfg.fixed_asset_pos_noise, device=self.device)
        self.init_fixed_pos_obs_noise[env_ids] = fixed_asset_pos_noise @ torch.diag(fixed_asset_pos_rand)

        # 3. Capture the post-event state for finite differencing
        # This ensures the first velocity calculation is zero.
        
        # 4. Update the "Safety Box" center based on the new asset location
        fixed_pos = self._fixed_asset.data.root_pos_w[env_ids] - self._env.scene.env_origins[env_ids]
        fixed_quat = self._fixed_asset.data.root_quat_w[env_ids]
        fixed_tip_pos_local = torch.zeros((len(env_ids), 3), device=self.device)
        fixed_tip_pos_local[:, 2] += self._env.cfg.task.fixed_asset_cfg.height + self._env.cfg.task.fixed_asset_cfg.base_height
        
        if self._env.cfg.task.name == "gear_mesh":
            fixed_tip_pos_local[:, 0] = self.cfg_task.fixed_asset_cfg.medium_gear_base_offset[0]
        _, fixed_tip_pos = torch_utils.tf_combine(
            fixed_quat, fixed_pos,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(len(env_ids), 1),
            fixed_tip_pos_local
        )
        self.fixed_pos_obs_frame[env_ids] = fixed_tip_pos

        self._compute_intermediate_values(dt=self._env.physics_dt)

        # 5. Sync velocity buffers to zero
        self.prev_fingertip_pos[env_ids] = self.fingertip_midpoint_pos[env_ids].clone()
        self.prev_fingertip_quat[env_ids] = self.fingertip_midpoint_quat[env_ids].clone()
        self.prev_joint_pos[env_ids] = self.joint_pos[env_ids, 0:7].clone()
        self.ee_linvel_fd[env_ids] = 0.0
        self.ee_angvel_fd[env_ids] = 0.0

         # 6) initialize control targets to current pose (prevents reset jerk)
        self.ctrl_target_fingertip_midpoint_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]
        self.ctrl_target_fingertip_midpoint_quat[env_ids] = self.fingertip_midpoint_quat[env_ids]

        # 7) make timestamp consistent
        self.last_update_timestamp = self._asset._data._sim_timestamp
    def _compute_intermediate_values(self, dt: float):
        """
        Compute intermediate values from raw tensors (matching DirectRLEnv).
        This includes finite-differencing for more reliable velocity estimates.
        """
        # Fixed asset state
        self.fixed_pos[:] = self._fixed_asset.data.root_pos_w - self._env.scene.env_origins
        self.fixed_quat[:] = self._fixed_asset.data.root_quat_w

        # Held asset state
        self.held_pos[:] = self._held_asset.data.root_pos_w - self._env.scene.env_origins
        self.held_quat[:] = self._held_asset.data.root_quat_w

        # Fingertip state
        self.fingertip_midpoint_pos[:] = self._asset.data.body_pos_w[:, self.fingertip_body_idx] - self._env.scene.env_origins
        self.fingertip_midpoint_quat[:] = self._asset.data.body_quat_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_linvel[:] = self._asset.data.body_lin_vel_w[:, self.fingertip_body_idx]
        self.fingertip_midpoint_angvel[:] = self._asset.data.body_ang_vel_w[:, self.fingertip_body_idx]

        # Jacobian
        jacobians = self._asset.root_physx_view.get_jacobians()
        left_finger_jacobian = jacobians[:, self.left_finger_body_idx - 1, 0:6, 0:7]
        right_finger_jacobian = jacobians[:, self.right_finger_body_idx - 1, 0:6, 0:7]
        self.fingertip_midpoint_jacobian[:] = (left_finger_jacobian + right_finger_jacobian) * 0.5
        
        # Mass matrix
        self.arm_mass_matrix[:] = self._asset.root_physx_view.get_generalized_mass_matrices()[:, 0:7, 0:7]
        
        # Joint states
        self.joint_pos[:] = self._asset.data.joint_pos.clone()
        self.joint_vel[:] = self._asset.data.joint_vel.clone()

        # Finite-differencing for more reliable velocity estimates (matching DirectRLEnv)
        self.ee_linvel_fd[:] = (self.fingertip_midpoint_pos - self.prev_fingertip_pos) / dt
        self.prev_fingertip_pos[:] = self.fingertip_midpoint_pos.clone()

        # Rotational velocity via finite differencing
        rot_diff_quat = torch_utils.quat_mul(
            self.fingertip_midpoint_quat, torch_utils.quat_conjugate(self.prev_fingertip_quat)
        )
        rot_diff_quat *= torch.sign(rot_diff_quat[:, 0]).unsqueeze(-1)
        rot_diff_aa = math_utils.axis_angle_from_quat(rot_diff_quat)
        self.ee_angvel_fd[:] = rot_diff_aa / dt
        self.prev_fingertip_quat[:] = self.fingertip_midpoint_quat.clone()

        # Joint velocity via finite differencing
        joint_diff = self.joint_pos[:, 0:7] - self.prev_joint_pos
        self.joint_vel_fd[:] = joint_diff / dt
        self.prev_joint_pos[:] = self.joint_pos[:, 0:7].clone()

        self.last_update_timestamp = self._asset._data._sim_timestamp

    def process_actions(self, actions: torch.Tensor):
        """Apply EMA smoothing and compute target poses (matching DirectRLEnv _pre_physics_step + _apply_action)."""
        self._raw_actions[:] = actions
        
        self.prev_actions[:] = self.actions.clone()
        # Apply EMA smoothing (matching DirectRLEnv _pre_physics_step)
        self.actions[:] = self.ema_factor * actions + (1.0 - self.ema_factor) * self.actions
        
        self._processed_actions[:] = self.actions  # Store smoothed actions for logging
        # Compute intermediate values if needed (matching DirectRLEnv)
        # Check if we need to re-compute velocities
        if self.last_update_timestamp < self._asset._data._sim_timestamp:
            dt = self._env.physics_dt
            self._compute_intermediate_values(dt=dt)
        
        # --- 1. Interpret Actions (matching DirectRLEnv _apply_action) ---
        # Pos: delta * threshold
        pos_actions = self.actions[:, 0:3] * self.pos_threshold
        # Rot: delta * threshold (with unidirectional check)
        rot_actions = self.actions[:, 3:6].clone()
        if self.unidirectional_rot:
            # Map [-1, 1] -> [-1, 0] for the Z-rotation axis (for nut_thread task)
            rot_actions[:, 2] = -(rot_actions[:, 2] + 1.0) * 0.5
        rot_actions = rot_actions * self.rot_threshold

        # --- 2. Compute Desired Position (with Clipping) ---
        # Initial Target
        ctrl_target_fingertip_midpoint_pos = self.fingertip_midpoint_pos + pos_actions
        
        # Get "Fixed Asset Observation Frame" (Fixed Asset Pos + Noise)
        # Matching DirectRLEnv: fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        fixed_pos_action_frame = self.fixed_pos_obs_frame + self.init_fixed_pos_obs_noise
        
        # Compute delta from fixed frame
        delta_pos = ctrl_target_fingertip_midpoint_pos - fixed_pos_action_frame
        # print("delta pos", delta_pos)
        # print("pos actoin", pos_actions)
        # Clip delta (Safety box around fixed asset) - using cfg.pos_action_bounds
        pos_error_clipped = torch.clip(
            delta_pos, 
            -torch.tensor(self.pos_action_bounds[0], device=self.device),
            torch.tensor(self.pos_action_bounds[1], device=self.device)
        )
        # Reconstruct valid target
        self.ctrl_target_fingertip_midpoint_pos[:] = fixed_pos_action_frame + pos_error_clipped

        # --- 3. Compute Desired Orientation (with Euler Constraint) ---
        # Convert axis-angle action to quat
        angle = torch.norm(rot_actions, p=2, dim=-1)
        axis = rot_actions / (angle.unsqueeze(-1) + 1e-6)
        
        rot_actions_quat = torch_utils.quat_from_angle_axis(angle, axis)
        # Handle zero-rotation case (matching DirectRLEnv)
        rot_actions_quat = torch.where(
            angle.unsqueeze(-1).repeat(1, 4) > 1e-6,
            rot_actions_quat,
            torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).repeat(self.num_envs, 1),
        )
        
        # Apply delta
        ctrl_target_fingertip_midpoint_quat = torch_utils.quat_mul(rot_actions_quat, self.fingertip_midpoint_quat)
        
        # Enforce "Upright" Constraint (Roll ~ PI, Pitch ~ 0)
        target_euler_xyz = torch.stack(torch_utils.get_euler_xyz(ctrl_target_fingertip_midpoint_quat), dim=1)
        target_euler_xyz[:, 0] = 3.14159  # Restrict actions to be upright
        target_euler_xyz[:, 1] = 0.0
        
        self.ctrl_target_fingertip_midpoint_quat[:] = torch_utils.quat_from_euler_xyz(
            roll=target_euler_xyz[:, 0], pitch=target_euler_xyz[:, 1], yaw=target_euler_xyz[:, 2]
        )
        

    def apply_actions(self):
        """Generate control signals using the exact DirectRLEnv compute_dof_torque function."""
        # Ensure intermediate values are up to date
        if self.last_update_timestamp < self._asset._data._sim_timestamp:
            self._compute_intermediate_values(dt=self._env.physics_dt)
        # Create a cfg-like object for compute_dof_torque
        # (it needs cfg.scene.num_envs, cfg.*)
        
        # Use the exact DirectRLEnv compute_dof_torque function
        joint_torque, applied_wrench = factory_control.compute_dof_torque(
            cfg=self._ctrl_cfg,  # Pass the env config (contains ctrl, scene, etc.)
            dof_pos=self.joint_pos,
            dof_vel=self.joint_vel,
            fingertip_midpoint_pos=self.fingertip_midpoint_pos,
            fingertip_midpoint_quat=self.fingertip_midpoint_quat,
            fingertip_midpoint_linvel=self.fingertip_midpoint_linvel,
            fingertip_midpoint_angvel=self.fingertip_midpoint_angvel,
            jacobian=self.fingertip_midpoint_jacobian,
            arm_mass_matrix=self.arm_mass_matrix,
            ctrl_target_fingertip_midpoint_pos=self.ctrl_target_fingertip_midpoint_pos,
            ctrl_target_fingertip_midpoint_quat=self.ctrl_target_fingertip_midpoint_quat,
            task_prop_gains=self.task_prop_gains,
            task_deriv_gains=self.task_deriv_gains,
            device=self.device,
            dead_zone_thresholds=self.dead_zone_thresholds,
        )
        # Set gripper position target (matching DirectRLEnv generate_ctrl_signals)
        self.ctrl_target_joint_pos[:] = self.joint_pos
        self.ctrl_target_joint_pos[:, 7:9] = 0.0  # Close gripper
        
        # Zero out gripper torques (position controlled)
        joint_torque[:, 7:9] = 0.0
        
        # Apply commands
        self._asset.set_joint_position_target(self.ctrl_target_joint_pos)
        self._asset.set_joint_effort_target(joint_torque)



# =============================================================================
# 2. Configuration
# =============================================================================

@configclass
class FactoryTaskSpaceControlCfg(ActionTermCfg):
    """
    Configuration for the Factory Task Space Control action.
    Matches DirectRLEnv's CtrlCfg structure exactly.
    """
    
    class_type = FactoryTaskSpaceControl
    
    # Asset names
    asset_name: str = "robot"
    joint_names: list[str] | str = ["panda_joint.*"]
    body_name: str = "panda_fingertip_centered"
    fixed_asset_name: str = "fixed_asset"
    held_asset_name: str = "held_asset"
    
    # EMA smoothing (matching DirectRLEnv cfg.ema_factor)
    ema_factor: float = 0.2
    
    # Action bounds (matching DirectRLEnv cfg)
    pos_action_bounds: list[float] = (0.05, 0.05)
    rot_action_bounds: list[float] = [1.0, 1.0, 1.0]
    
    # Action thresholds (matching DirectRLEnv cfg)
    pos_action_threshold: list[float] = [0.02, 0.02, 0.02]
    rot_action_threshold: list[float] = [0.097, 0.097, 0.097]
    
    # Reset gains (matching DirectRLEnv cfg)
    reset_joints: list[float] = [1.5178e-03, -1.9651e-01, -1.4364e-03, -1.9761, -2.7717e-04, 1.7796, 7.8556e-01]
    reset_task_prop_gains: list[float] = [300, 300, 300, 20, 20, 20]
    reset_rot_deriv_scale: float = 10.0
    
    # Default task gains (matching DirectRLEnv cfg)
    default_task_prop_gains: list[float] = [100, 100, 100, 30, 30, 30]
    
    # Nullspace control parameters (matching DirectRLEnv cfg)
    default_dof_pos_tensor: list[float] = [-1.3003, -0.4015, 1.1791, -2.1493, 0.4001, 1.9425, 0.4754]
    kp_null: float = 10.0
    kd_null: float = 6.3246
    
    # Observation randomization (matching DirectRLEnv cfg.obs_rand)
    fixed_asset_pos_noise: list[float] = [0.001, 0.001, 0.001]
    
    # Task-specific settings
    unidirectional_rot: bool = False  # Set to True for nut_thread task
    
    # For compatibility with compute_dof_torque (needs to access scene.num_envs)
    # This will be set at runtime but we need a placeholder
    scene: object = None  # Will be populated with a reference to env.scene