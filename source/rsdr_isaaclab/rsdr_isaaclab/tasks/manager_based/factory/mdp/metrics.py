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

    env.extras["success_instant"] = curr_successes.float().mean()
    first_success = curr_successes & ~env.ep_succeeded
    env.ep_succeeded |= curr_successes
    fs_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
    if fs_ids.numel() > 0:
        env.ep_success_times[fs_ids] = env.episode_length_buf[fs_ids]
    reset_ids = env_ids    
    if reset_ids.numel() > 0:
        # episode success (latched)
        env.extras["success"] = env.ep_succeeded[reset_ids].float().mean()

        succ_mask = env.ep_succeeded[reset_ids]
        if succ_mask.any():
            env.extras["success_times"] = env.ep_success_times[reset_ids[succ_mask]].float().mean()

        # reset trackers
        env.ep_succeeded[reset_ids] = False
        env.ep_success_times[reset_ids] = 0
    return log_factory_metrics_direct_style(env, env_ids, curr_successes)

def log_factory_metrics_direct_style(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,  # ignore unless you KNOW it's reset ids
    curr_successes: torch.Tensor,  # [num_envs] bool
):
    # Buffers (match Direct semantics; dtype can be long)
    if not hasattr(env, "ep_succeeded"):
        env.ep_succeeded = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)
        env.ep_success_times = torch.zeros((env.num_envs,), dtype=torch.long, device=env.device)

    # --- Direct: first_success = curr_successes & ~ep_succeeded ---
    first_success = curr_successes & torch.logical_not(env.ep_succeeded.bool())
    env.ep_succeeded[curr_successes] = 1

    first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
    if first_success_ids.numel() > 0:
        env.ep_success_times[first_success_ids] = env.episode_length_buf[first_success_ids]

    nonzero_success_ids = env.ep_success_times.nonzero(as_tuple=False).squeeze(-1)
    if nonzero_success_ids.numel() > 0:
        env.extras["success_times"] = env.ep_success_times[nonzero_success_ids].sum() / nonzero_success_ids.numel()

    done = env.termination_manager.terminated.clone()
    for attr in ["time_outs", "timeouts", "truncated", "truncations"]:
        if hasattr(env.termination_manager, attr):
            done |= getattr(env.termination_manager, attr)

    env.extras["debug/done_sum"] = done.float().sum()
    env.extras["debug/done_is_all_or_none"] = ((done.all() | (~done).all()).float())
    # If your env is sync-reset, done is either all true or all false
    reset_ids = env_ids
    if reset_ids.numel() == env.num_envs:
        env.extras["successes"] = torch.count_nonzero(curr_successes) / env.num_envs
    else:
        # per-env reset: closest analog
        env.extras["successes"] = torch.count_nonzero(curr_successes[reset_ids]) / reset_ids.numel()

    # Reset buffers for those envs (Direct does this in _pre_physics_step)
    env.ep_succeeded[reset_ids] = 0
    env.ep_success_times[reset_ids] = 0

    return env.extras.get("successes", torch.tensor(0.0, device=env.device))
import torch
def log_episode_returns(env, env_ids: torch.Tensor):
    device = env.device

    # During the very first reset before stepping, reward_manager may exist but sums are zero -> fine.
    if not hasattr(env, "reward_manager") or not hasattr(env.reward_manager, "_episode_sums"):
        env.extras["episode_return"] = torch.tensor(0.0, device=device)
        return env.extras["episode_return"]

    # total episodic return per env = sum over terms of RewardManager._episode_sums[term]
    total = torch.zeros(env.num_envs, device=device)
    for v in env.reward_manager._episode_sums.values():
        total += v

    if env_ids.numel() > 0:
        env.extras["episode_return"] = total[env_ids].mean()
    else:
        # no reset happening right now
        env.extras.setdefault("episode_return", torch.tensor(0.0, device=device))

    return env.extras["episode_return"]


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

    return grasp_dist.mean()

# def log_reward_components(env, env_ids):
#     """Log individual reward components to see which one is non-zero."""
#     # We look at the extras where we stored them in our reward function
#     if "log" in env.extras:
#         for key in ["kp_baseline", "kp_coarse", "kp_fine"]:
#             if key in env.extras["log"]:
#                  env.extras[f"debug/rew_{key}"] = env.extras["log"][key].mean()
#     return torch.tensor(0.0, device=env.device)


def check_first_frame_stats(env: ManagerBaFactoryEnvsedRLEnv, env_ids: torch.Tensor):
    """
    Sanity check that runs ONLY on the first step of an episode.
    Prints warnings if spawn velocity is high (explosion) or if spawn variance is zero (no noise).
    """
    # 1. Filter for environments that are on their very first step (time=0)
    # We use episode_length_buf which is 0 right after a reset
    new_ids = (env.episode_length_buf[env_ids] == 0).nonzero(as_tuple=False).squeeze(-1)
    
    if len(new_ids) > 0:
        robot = env.scene["robot"]
        
        # --- A. Check for "Explosions" (High Velocity) ---
        # Get fingertip velocity
        idx = robot.body_names.index("panda_fingertip_centered")
        vel = robot.data.body_lin_vel_w[new_ids, idx]
        speed = torch.norm(vel, dim=-1)
        
        max_speed = speed.max().item()
        if max_speed > 0.1: # Threshold: 10cm/s
            print(f"[WARNING] Unstable Reset! Max Velocity on Frame 0: {max_speed:.4f} m/s")
        
        # --- B. Check for "Broken Noise" (Zero Variance) ---
        # We only check this if we have enough envs to calculate a meaningful std dev
        if len(new_ids) > 10:
            # Get fingertip positions
            pos = robot.data.body_pos_w[new_ids, idx]
            
            # Calculate Standard Deviation across the batch
            std_dev = pos.std(dim=0) # Returns [std_x, std_y, std_z]
            
            # If standard deviation is extremely low, randomization is missing
            if torch.max(std_dev) < 1e-4:
                print(f"[CRITICAL] No Randomization Detected! Std Dev: {std_dev.tolist()}")
            else:
                # Optional: Print confirmed noise levels to verify config
                # print(f"[INFO] Spawn Distribution Std Dev: {std_dev.tolist()}")
                pass

    # Curriculum terms must return a tensor, even if unused
    return torch.tensor(0.0, device=env.device)
def unpack_obs_group(env, group_name="critic"):
    """
    Slices the concatenated observation tensor back into a dictionary 
    using the Manager's metadata.
    """
    # 1. Get the full concatenated tensor
    obs_tensor = env.observation_manager.compute()[group_name]
    
    # 2. Get the metadata
    term_names = env.observation_manager.active_terms[group_name]
    term_shapes = env.observation_manager.group_obs_term_dim[group_name]
    
    # 3. Iterate and Slice
    obs_dict = {}
    current_idx = 0
    
    for name, shape in zip(term_names, term_shapes):
        # shape is (num_envs, dim1, dim2...), we care about the flattened width
        # But wait! ManagerBasedRLEnv flattens history/dims before concatenating.
        # We need the last dimension size.
        
        # Calculate width of this term in the 1D vector
        term_width = 1
        for dim in shape:
            term_width *= dim
            
        # For num_envs, the shape usually excludes the batch dim in this list, 
        # checking the docs: "value is a list of tuples representing the shape... for a term"
        # Usually it is (dim,) for a vector.
        
        # Slice
        obs_dict[name] = obs_tensor[:, current_idx : current_idx + term_width]
        current_idx += term_width
        
    return obs_dict

def debug_observation_freshness(env, env_ids):
    """
    Checks if observations are stale by teleporting the robot and 
    verifying if the observation updates immediately.
    """
    # 1. Reset to get clean state
    # We do a 'soft' reset of just the robot joints to avoid triggering the full expensive Reset Logic
    robot = env.scene["robot"]
    env_id = 0
    
    # 2. Define Test Joint Position (1.5 rad on all joints)
    test_joint_pos = robot.data.default_joint_pos[env_id].clone() + 0.5
    # Zero out gripper joints (last 2 usually) if needed, but 1.5 is fine for debug
    
    # 3. Teleport: Write directly to Simulation (Bypassing Action Manager)
    robot.write_joint_state_to_sim(
        position=test_joint_pos, 
        velocity=torch.zeros_like(test_joint_pos), 
        env_ids=torch.tensor([env_id], device=env.device)
    )
    
    # 4. Compute and Unpack
    print("Computing Observations...")
    # This calls compute() internally
    unpacked_critic = unpack_obs_group(env, "critic")
    
    # 5. Access by Name safely
    if "joint_pos" in unpacked_critic:
        obs_joint_pos = unpacked_critic["joint_pos"][0] # Env 0
        
        # Verify
        error = torch.norm(obs_joint_pos[:7] - test_joint_pos[:7])
        print(f"Error Norm: {error.item():.6f}")
        
        if error < 1e-3:
            print("✅ PASS: Observations are FRESH.")
        else:
            print("❌ FAIL: Observations are STALE.")
    else:
        print(f"Keys found: {list(unpacked_critic.keys())}")
        print("❌ FAIL: 'joint_pos' not found in Critic observations.")

    return torch.tensor(0.0, device=env.device)

import torch
import isaacsim.core.utils.torch as torch_utils

def debug_gear_mesh_fixed_reference(env, env_ids, fixed_asset_cfg, task_cfg, obs_group="policy",
                                   rel_key="fingertip_pos_rel_fixed"):
    dev = env.device
    N = env.num_envs
    o = env.scene.env_origins

    # 1) fingertip in env frame
    robot = env.scene["robot"]
    ft_idx = robot.body_names.index("panda_fingertip_centered")
    ft_e = robot.data.body_pos_w[:, ft_idx] - o

    # 2) infer what fixed reference your OBS is using
    obs = env.observation_manager.compute()[obs_group]  # concatenated
    # If you have unpack_obs_group, use it instead:
    # obs_dict = unpack_obs_group(env, obs_group)
    # rel = obs_dict[rel_key]

    # --- minimal slice approach if you know rel_key exists as a term ---
    obs_dict = unpack_obs_group(env, obs_group)
    if rel_key not in obs_dict:
        env.extras["dbg/missing_rel_key"] = torch.tensor(1.0, device=dev)
        return torch.tensor(0.0, device=dev)

    rel = obs_dict[rel_key]  # [N,3] (or wider; then take first 3)
    rel = rel[:, :3]

    fixed_ref_inferred = ft_e - rel

    # 3) compute Direct-style fixed_pos_obs_frame (env frame)
    fixed = env.scene[fixed_asset_cfg.name]
    fixed_pos_e = fixed.data.root_pos_w - o
    fixed_quat = fixed.data.root_quat_w

    tip_local = torch.zeros((N, 3), device=dev)
    tip_local[:, 2] = task_cfg.fixed_asset_cfg.height + task_cfg.fixed_asset_cfg.base_height
    if task_cfg.name == "gear_mesh":
        tip_local[:, 0] = task_cfg.fixed_asset_cfg.medium_gear_base_offset[0]

    I = torch.tensor([1.0, 0.0, 0.0, 0.0], device=dev).repeat(N, 1)
    _, fixed_obs_e = torch_utils.tf_combine(fixed_quat, fixed_pos_e, I, tip_local)

    # 4) log error (mean over envs)
    err = torch.norm(fixed_ref_inferred - fixed_obs_e, dim=1).mean()
    env.extras["dbg/fixed_ref_err"] = err

    # Optional: just for gear_mesh, watch the xy error too
    env.extras["dbg/fixed_ref_xy_err"] = torch.norm(
        (fixed_ref_inferred - fixed_obs_e)[:, :2], dim=1
    ).mean()
    err_vec = fixed_ref_inferred - fixed_obs_e     # [N,3]
    env.extras["dbg/fixed_ref_err_x"] = err_vec[:,0].abs().mean()
    env.extras["dbg/fixed_ref_err_y"] = err_vec[:,1].abs().mean()
    env.extras["dbg/fixed_ref_err_z"] = err_vec[:,2].abs().mean()
    off = task_cfg.fixed_asset_cfg.medium_gear_base_offset[0] if task_cfg.name == "gear_mesh" else 0.0

    local = torch.tensor([off, 0.0, 0.0], device=env.device).repeat(env.num_envs, 1)
    rot = torch_utils.quat_apply(fixed_quat, local)   # this is the correct “x offset in local frame”
    env.extras["dbg/rot_off_norm"] = rot.norm(dim=1).mean()
    mask = (env.episode_length_buf > 0)
    return err

# import torch
# from isaaclab.envs import ManagerBasedRLEnv
# from isaaclab.managers import SceneEntityCfg
# from .. import factory_utils  # Ensure you have your utils imported

# def log_factory_statistics(
#     env: ManagerBasedRLEnv,
#     env_ids: torch.Tensor,
#     held_asset_cfg: SceneEntityCfg,
#     fixed_asset_cfg: SceneEntityCfg,
#     task_cfg: object, # Pass the Task Config object (PegInsert, etc.)
# ):
#     """
#     Replicates FactoryEnv._log_factory_metrics completely.
#     Tracks:
#     1. Instantaneous Success (curr_successes)
#     2. Latched Success (ep_succeeded) - Did it succeed at least once this episode?
#     3. Time to First Success (success_times)
#     4. Success Rate on Reset
#     """
#     # --- 1. State Initialization (Monkey-Patching) ---
#     # We attach buffers to the env object if they don't exist yet
#     if not hasattr(env, "factory_ep_succeeded"):
#         env.factory_ep_succeeded = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
#         env.factory_ep_success_times = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)

#     # --- 2. Calculate Current Success ---
#     # (Replicating _get_curr_successes logic)
#     held_asset = env.scene[held_asset_cfg.name]
#     fixed_asset = env.scene[fixed_asset_cfg.name]
    
#     held_pos = held_asset.data.root_pos_w - env.scene.env_origins
#     held_quat = held_asset.data.root_quat_w
#     fixed_pos = fixed_asset.data.root_pos_w - env.scene.env_origins
#     fixed_quat = fixed_asset.data.root_quat_w

#     held_base_pos, _ = factory_utils.get_held_base_pose(
#         held_pos, held_quat, task_cfg.name, task_cfg.fixed_asset_cfg, env.num_envs, env.device
#     )
#     target_held_base_pos, _ = factory_utils.get_target_held_base_pose(
#         fixed_pos, fixed_quat, task_cfg.name, task_cfg.fixed_asset_cfg, env.num_envs, env.device
#     )

#     xy_dist = torch.linalg.vector_norm(target_held_base_pos[:, 0:2] - held_base_pos[:, 0:2], dim=1)
#     z_disp = held_base_pos[:, 2] - target_held_base_pos[:, 2]

#     # Success Threshold Checks
#     is_centered = xy_dist < 0.0025
    
#     # Handle task-specific height thresholds
#     fixed_cfg = task_cfg.fixed_asset_cfg
#     if task_cfg.name == "nut_thread":
#         height_thresh = fixed_cfg.thread_pitch * task_cfg.success_threshold
#     else: # Peg / Gear
#         height_thresh = fixed_cfg.height * task_cfg.success_threshold
        
#     is_close_or_below = z_disp < height_thresh
#     curr_successes = is_centered & is_close_or_below

#     # [Nut Thread] Add Rotation Check
#     if task_cfg.name == "nut_thread":
#         # Check yaw angle
#         robot = env.scene["robot"]
#         # Assuming you have the body index stored or find it
#         fingertip_body_idx = robot.body_names.index("panda_fingertip_centered")
            
#         fingertip_quat = robot.data.body_quat_w[:, fingertip_body_idx]
#         from isaacsim.core.utils.torch import get_euler_xyz
#         _, _, curr_yaw = get_euler_xyz(fingertip_quat)
#         curr_yaw = factory_utils.wrap_yaw(curr_yaw)
#         is_rotated = curr_yaw < task_cfg.ee_success_yaw
#         curr_successes = curr_successes & is_rotated

#     # --- 3. Update Latched Statistics ---
#     # Identify environments that just succeeded for the first time this episode
#     first_success = curr_successes & ~env.factory_ep_succeeded
    
#     # Mark them as succeeded
#     env.factory_ep_succeeded[curr_successes] = True
    
#     # Record the time (step count) of the first success
#     # Note: episode_length_buf tracks steps since reset
#     first_success_ids = first_success.nonzero(as_tuple=False).squeeze(-1)
#     if len(first_success_ids) > 0:
#         env.factory_ep_success_times[first_success_ids] = env.episode_length_buf[first_success_ids].float()

#     # --- 4. Logging on Reset ---
#     # We check the reset buffer to find episodes that are finishing NOW
#     # reset_buf is usually populated by the TerminationManager before this runs
#     reset_ids = env_ids
    
#     if len(reset_ids) > 0:
#         # A. Log Success Rate (Did they succeed at any point?)
#         success_rate = env.factory_ep_succeeded[reset_ids].float().mean()
#         env.extras["successes"] = success_rate

#         # B. Log Success Times (Only for those that succeeded)
#         # We filter the reset IDs to find which ones were successful
#         succeeded_mask = env.factory_ep_succeeded[reset_ids]
#         if succeeded_mask.any():
#             avg_time = env.factory_ep_success_times[reset_ids[succeeded_mask]].mean()
#             env.extras["success_times"] = avg_time

#         # C. Reset Buffers for next episode
#         env.factory_ep_succeeded[reset_ids] = False
#         env.factory_ep_success_times[reset_ids] = 0.0

#     # --- 5. Return Scalar (Required for CurriculumSystem) ---
#     return env.extras.get("successes", torch.tensor(0.0, device=env.device))