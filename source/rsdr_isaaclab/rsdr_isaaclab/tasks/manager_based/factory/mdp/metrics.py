# rsdr_isaaclab/tasks/manager_based/factory/mdp/metrics.py

from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Any

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from ..factory_utils import get_held_base_pose, get_target_held_base_pose, wrap_yaw
import isaacsim.core.utils.torch as torch_utils

def log_factory_success_metrics(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    held_asset_cfg: SceneEntityCfg,
    fixed_asset_cfg: SceneEntityCfg,
    factory_task_cfg: Any,
):
    # 1. Initialize buffers on first call
    if not hasattr(env, "ep_succeeded"):
        env.ep_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        env.ep_success_times = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)

    # 2. Get Asset Poses and Success Mask
    held_asset = env.scene[held_asset_cfg.name]
    fixed_asset = env.scene[fixed_asset_cfg.name]
    
    held_pos = held_asset.data.root_pos_w - env.scene.env_origins
    held_quat = held_asset.data.root_quat_w
    fixed_pos = fixed_asset.data.root_pos_w - env.scene.env_origins
    fixed_quat = fixed_asset.data.root_quat_w

    held_base_pos, _ = get_held_base_pose(held_pos, held_quat, factory_task_cfg.name, factory_task_cfg.fixed_asset_cfg, env.num_envs, env.device)
    target_held_base_pos, _ = get_target_held_base_pose(fixed_pos, fixed_quat, factory_task_cfg.name, factory_task_cfg.fixed_asset_cfg, env.num_envs, env.device)

    xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
    z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]
    
    fixed_cfg = factory_task_cfg.fixed_asset_cfg
    height_threshold = (fixed_cfg.height * factory_task_cfg.success_threshold 
                        if factory_task_cfg.name != "nut_thread" 
                        else fixed_cfg.thread_pitch * factory_task_cfg.success_threshold)
    
    curr_successes = (xy_dist < 0.0025) & (z_disp < height_threshold)

    # [NUT THREAD] Add rotation check
    if factory_task_cfg.name == "nut_thread":
        fingertip_quat = env.scene["robot"].data.body_quat_w[:, env.scene["robot"].body_names.index("panda_fingertip_centered")]
        _, _, curr_yaw = torch_utils.get_euler_xyz(fingertip_quat)
        curr_yaw = wrap_yaw(curr_yaw)
        curr_successes &= (curr_yaw < factory_task_cfg.ee_success_yaw)

    # 3. Handle Logging on Reset (Sync with RL Games 'success' key)
    reset_ids = env.termination_manager.terminated.nonzero(as_tuple=False).squeeze(-1)
    
    if len(reset_ids) > 0:
        # RL Games looks for 'success' in env.extras
        env.extras["success"] = curr_successes.float().mean()
        
        # Log Reward Penalties for debugging
        # We access the current actions from the ActionManager
        actions = env.action_manager.action
        env.extras["action_penalty"] = torch.norm(actions, p=2, dim=-1).mean()
        
        # Reset tracking for these envs
        env.ep_succeeded[reset_ids] = False
        env.ep_success_times[reset_ids] = 0

    # 4. Success Timing logic
    first_success = curr_successes & ~env.ep_succeeded
    env.ep_succeeded[curr_successes] = True
    
    first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
    if len(first_success_ids) > 0:
        env.ep_success_times[first_success_ids] = env.episode_length_buf[first_success_ids]

    # 5. Average success time
    nonzero_success_ids = env.ep_success_times.nonzero(as_tuple=False).squeeze(-1)
    if len(nonzero_success_ids) > 0:
        env.extras["success_times"] = env.ep_success_times[nonzero_success_ids].float().mean()

    return env.extras.get("success", torch.tensor(0.0, device=env.device))

def log_grasp_stability(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg,
    held_asset_cfg: SceneEntityCfg,
):
    """Tracks how far the asset is from the center of the fingers."""
    robot = env.scene[robot_cfg.name]
    held_asset = env.scene[held_asset_cfg.name]

    # 1. Get current world positions
    fingertip_pos = robot.data.body_pos_w[:, robot.body_names.index("panda_fingertip_centered")]
    held_pos = held_asset.data.root_pos_w

    # 2. Compute Euclidean distance
    grasp_dist = torch.norm(fingertip_pos - held_pos, dim=-1)

    # 3. Log the mean distance to extras for TensorBoard
    # If this value increases significantly during an episode, the object dropped.
    env.extras["grasp_error_avg"] = grasp_dist.mean()
    
    # 4. Binary check: Did we lose the object? (Threshold: 5cm)
    is_dropped = (grasp_dist > 0.05).float().mean()
    env.extras["drop_rate"] = is_dropped

    return grasp_dist