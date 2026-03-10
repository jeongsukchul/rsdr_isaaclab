from __future__ import annotations

import torch


def _get_articulation(env, name: str):
    if hasattr(env, "scene") and hasattr(env.scene, "articulations") and name in env.scene.articulations:
        return env.scene.articulations[name]
    attr = f"_{name}"
    if hasattr(env, attr):
        return getattr(env, attr)
    raise KeyError(f"Could not find articulation '{name}' on env.")


def _extract_param(sampler, master_values: torch.Tensor, param_name: str, device, default_dim: int = 1):
    """Return sampled slices for param_name. Falls back to zeros when missing."""
    for p in sampler.cfg.params:
        if p.name == param_name:
            return master_values[:, p.indices]
    return torch.zeros((master_values.shape[0], default_dim), device=device, dtype=master_values.dtype)


def randomize_mass_scale(env, env_ids, asset_name: str, values: torch.Tensor, body_ids=None):
    asset = _get_articulation(env, asset_name)
    env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    if body_ids is None:
        body_ids = torch.arange(asset.num_bodies, device="cpu", dtype=torch.long)
    else:
        body_ids = body_ids.to(device="cpu", dtype=torch.long)

    masses = asset.root_physx_view.get_masses().clone().to(device="cpu")
    default_mass = asset.data.default_mass.detach().to(device="cpu")
    default = default_mass[env_ids_cpu[:, None], body_ids]
    values = values.to(device="cpu", dtype=default.dtype)

    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if values.shape[1] == 1:
        values = values.repeat(1, default.shape[1])
    elif values.shape[1] != default.shape[1]:
        values = values.mean(dim=1, keepdim=True).repeat(1, default.shape[1])

    masses[env_ids_cpu[:, None], body_ids] = default * values
    asset.root_physx_view.set_masses(masses, env_ids_cpu)


def randomize_friction_scale(env, env_ids, asset_name: str, values: torch.Tensor):
    asset = _get_articulation(env, asset_name)
    env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    mats = asset.root_physx_view.get_material_properties().clone().to(device="cpu")
    values = values.to(device="cpu", dtype=mats.dtype)

    if values.dim() == 1:
        values = values.unsqueeze(-1)
    if values.shape[1] != 1:
        values = values.mean(dim=1, keepdim=True)

    vals = values.reshape(-1, 1)
    mats[env_ids_cpu, :, 0] = vals
    mats[env_ids_cpu, :, 1] = vals
    asset.root_physx_view.set_material_properties(mats, env_ids_cpu)


def apply_learned_randomization(env, env_ids: torch.Tensor | None = None):
    """Sample DR contexts, apply physics randomization, and store env buffers."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.device)

    n = int(env_ids.numel())
    if n == 0:
        return None

    if env._uniform_eval:
        sample_fn = env.sampler.get_test_sample_fn()
        sampled = sample_fn(n).detach()
        mapping = None
    else:
        sample_fn = env.sampler.get_train_sample_fn()
        if env.sampler.name == "GMMVI":
            sampled, mapping = sample_fn(n)
        else:
            sampled = sample_fn(n).detach()
            mapping = None

    sampled = sampled.to(device=env.device, dtype=torch.float32)
    env.dr_context[env_ids] = sampled
    if mapping is not None:
        env.mapping[env_ids] = mapping.to(device=env.device, dtype=torch.int32)

    for p_cfg in env.sampler.cfg.params:
        vals = sampled[:, p_cfg.indices]
        if p_cfg.event_type == "mass":
            randomize_mass_scale(env, env_ids, p_cfg.target_asset, vals, p_cfg.target_indices)
        elif p_cfg.event_type == "friction":
            randomize_friction_scale(env, env_ids, p_cfg.target_asset, vals)

    env.extras["dr_samples"] = env.dr_context
    return sampled
