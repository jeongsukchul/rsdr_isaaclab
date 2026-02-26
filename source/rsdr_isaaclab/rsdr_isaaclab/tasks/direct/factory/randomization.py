# rsdr_isaaclab/tasks/direct/factory/randomization.py
from __future__ import annotations

import torch
import carb
from isaaclab.sim import SimulationContext
from isaaclab.actuators import ImplicitActuator
from rsdr_isaaclab.tasks.direct.factory import factory_utils


def _get_articulation(env, name: str):
    # DirectRLEnv has env.scene.articulations dict
    if hasattr(env, "scene") and hasattr(env.scene, "articulations") and name in env.scene.articulations:
        return env.scene.articulations[name]
    # fallback (in case you pass "robot"/etc as attributes)
    attr = f"_{name}"
    if hasattr(env, attr):
        return getattr(env, attr)
    raise KeyError(f"Could not find articulation '{name}' on Direct env.")


def _extract_param(sampler, master_values: torch.Tensor, param_name: str, device, default_dim: int = 1):
    """Return master_values[:, p.indices] for param_name. Safe fallback to zeros."""
    for p in sampler.cfg.params:
        if p.name == param_name:
            return master_values[:, p.indices]
    return torch.zeros((master_values.shape[0], default_dim), device=device, dtype=master_values.dtype)


def _broadcast_multiplier(vals: torch.Tensor, target_width: int) -> torch.Tensor:
    """
    vals: (N, K)
    returns (N, target_width)
    Rules:
      - K==1: broadcast
      - K==target_width: use as-is
      - otherwise: use mean scalar and broadcast
    """
    if vals is None:
        return None
    if vals.dim() == 1:
        vals = vals.unsqueeze(-1)
    if vals.shape[1] == 1:
        return vals.repeat(1, target_width)
    if vals.shape[1] == target_width:
        return vals
    # fallback: scalar per env
    return vals.mean(dim=1, keepdim=True).repeat(1, target_width)


def randomize_actuator_gain(env, vals):
    """Multiply default joint stiffness/damping by per-env multipliers."""
    env.default_gains = torch.tensor(env.cfg.ctrl.default_task_prop_gains, device=env.device).repeat(
            (env.num_envs, 1)
        ) * vals

def randomize_mass(env, env_ids, asset_name: str, values: torch.Tensor, body_ids=None):
    """Set mass = default_mass * values for selected bodies."""
    asset = _get_articulation(env, asset_name)
    if body_ids is None:
        body_ids = torch.arange(asset.num_bodies, device=env.device)

    masses = asset.root_physx_view.get_masses().clone()
    default = asset.data.default_mass[env_ids[:, None], body_ids]  # (N, nbodies)

    # Broadcast values
    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if values.shape[1] == 1:
        values = values.repeat(1, default.shape[1])
    elif values.shape[1] != default.shape[1]:
        # fallback scalar mean
        values = values.mean(dim=1, keepdim=True).repeat(1, default.shape[1])

    masses[env_ids[:, None], body_ids] = default * values
    asset.root_physx_view.set_masses(masses, env_ids)
    

def apply_learned_randomization(env, env_ids=None):
    """
    DirectEnv reset hook:
      1) sample master_values
      2) apply physics DR (mass, actuator stiffness/damping)
      3) call env.randomize_initial_state(..., master_values=..., gravity_z=...)
    """
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    # Direct FactoryEnv reset assumes sync reset (your own comment in _get_dones)
    if len(env_ids) != env.num_envs:
        raise RuntimeError("Direct FactoryEnv reset path assumes sync reset (env_ids must be all envs).")

    sampler = env.sampler
    if sampler is None:
        # fallback to original behavior
        raise ValueError("No sampler found on env; cannot apply learned randomization.")
    print("reset with uniform dist? :", env._uniform_eval)
    if env._uniform_eval:
        sample_fn = env.sampler.get_test_sample_fn()
        master_values = sample_fn(env.num_envs).detach()
    else:
        if env.sampler.name == "GMMVI":
            sample_fn = env.sampler.get_train_sample_fn()
            master_values, mapping = sample_fn(env.num_envs)
            master_values = master_values.to(device=env.dr_context.device, dtype=env.dr_context.dtype)
            mapping = mapping.to(device=env.dr_context.device, dtype=env.dr_context.dtype)
            env.mapping  = mapping
        else:
            sample_fn = env.sampler.get_train_sample_fn()
            master_values = sample_fn(env.num_envs).detach()
            master_values = master_values.to(device=env.dr_context.device, dtype=env.dr_context.dtype)
    # log_probs = sampler.log_prob(master_values).to(env.device).detach()
    env.dr_context = master_values
    # store for training / debugging
    if "dr_samples" not in env.extras:
        env.extras["dr_samples"] = torch.zeros((env.num_envs, sampler.num_params), device=env.device)
        # env.extras["dr_log_probs"] = torch.zeros((env.num_envs,), device=env.device)

    env.extras["dr_samples"] = master_values
    # env.extras["dr_log_probs"][env_ids] = log_probs

    stiff_val = None
    damping_val = None
    gravity_val = None

    # Physics DR dispatch
    for p_cfg in sampler.cfg.params:
        vals = master_values[:, p_cfg.indices]
        if p_cfg.event_type == "stiffness":
            stiff_val = vals
        # elif p_cfg.event_type == "damping":
        #     damping_val = vals
        elif p_cfg.event_type == "mass":
            randomize_mass(env, env_ids, p_cfg.target_asset, vals, p_cfg.target_indices)
        elif p_cfg.event_type == "gravity":
            gravity_val = vals
        elif p_cfg.event_type == "robot_friction":
            factory_utils.set_friction(env._robot, vals, env.scene.num_envs)
        elif p_cfg.event_type == "fixed_friction":
            factory_utils.set_friction(env._fixed_asset, vals, env.scene.num_envs)
        elif p_cfg.event_type == "held_friction":
            factory_utils.set_friction(env._held_asset, vals, env.scene.num_envs)
        elif p_cfg.event_type == "pos_threshold":
            env.pos_thrsehold = torch.tensor(env.cfg.ctrl.pos_action_threshold, device=vals.device, dtype=vals.dtype) * vals
        elif p_cfg.event_type == "rot_threshold":
            env.rot_threshold =torch.tensor(env.cfg.ctrl.rot_action_threshold, device=vals.device, dtype=vals.dtype) * vals
        elif p_cfg.event_type == "ema":
            env.ema_factor = vals
    randomize_actuator_gain(env, stiff_val)
    # self.pos_threshold = forge_utils.get_random_prop_gains(
    #         self.pos_threshold, self.cfg.ctrl.pos_threshold_noise_level, self.num_envs, self.device
    #     )
    # self.rot_threshold = forge_utils.get_random_prop_gains(
    #         self.rot_threshold, self.cfg.ctrl.rot_threshold_noise_level, self.num_envs, self.device
    #     )
    # gravity_z = gravity_val.mean().item() if gravity_val is not None else None

    # Kinematics/state reset using the sampled pose noises
    env.randomize_initial_state(env_ids, master_values=master_values, sample_fn=sample_fn)
