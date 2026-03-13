"""
Code for the General Bridge Sampler (GBS).
Fur further details see: https://arxiv.org/abs/2307.01198
"""

from functools import partial
from time import time
from typing import Optional, Sequence

import distrax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
import matplotlib.pyplot as plt
from flax.training import train_state
from flax import linen as nn
from matplotlib.ticker import MaxNLocator

# IMPORTANT: gbs_loss must provide a sampler that does NOT need target logprob inside.
# I assume you expose: rnd_no_target(...) -> (x0, xT, log_ratio)
from .gbs_loss import (
    VP,
    rnd_no_target,
    rnd_time_reversal_lv_no_target,
    lv_loss_from_rnd,
)


class PISGRADNet(nn.Module):
    dim: int

    num_layers: int = 2
    num_hid: int = 64
    outer_clip: float = 1e4
    inner_clip: float = 1e2

    weight_init: float = 1e-8
    bias_init: float = 0.0

    def setup(self):
        self.timestep_phase = self.param(
            "timestep_phase", nn.initializers.zeros_init(), (1, self.num_hid)
        )
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.num_hid)[None]

        self.time_coder_state = nn.Sequential(
            [nn.Dense(self.num_hid), nn.gelu, nn.Dense(self.num_hid)]
        )

        self.time_coder_grad = nn.Sequential(
            [nn.Dense(self.num_hid)]
            + [
                nn.Sequential([nn.gelu, nn.Dense(self.num_hid)])
                for _ in range(self.num_layers)
            ]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(self.weight_init),
                    bias_init=nn.initializers.constant(self.bias_init),
                )
            ]
        )

        self.state_time_net = nn.Sequential(
            [nn.Sequential([nn.Dense(self.num_hid), nn.gelu]) for _ in range(self.num_layers)]
            + [
                nn.Dense(
                    self.dim,
                    kernel_init=nn.initializers.constant(1e-8),
                    bias_init=nn.initializers.zeros_init(),
                )
            ]
        )

    def get_fourier_features(self, timesteps):
        sin_embed_cond = jnp.sin((self.timestep_coeff * timesteps) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * timesteps) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    def __call__(self, input_array, time_array, lgv_term):
        time_array_emb = self.get_fourier_features(time_array)
        if len(input_array.shape) == 1:
            time_array_emb = time_array_emb[0]

        t_net1 = self.time_coder_state(time_array_emb)
        t_net2 = self.time_coder_grad(time_array_emb)

        extended_input = jnp.concatenate((input_array, t_net1), axis=-1)
        out_state = self.state_time_net(extended_input)
        out_state = jnp.clip(out_state, -self.outer_clip, self.outer_clip)

        lgv_term = jnp.clip(lgv_term, -self.inner_clip, self.inner_clip)
        out_state_p_grad = out_state + t_net2 * lgv_term
        return out_state_p_grad


def plot_sample_density_2d(
    samples,
    low,
    high,
    bins: int = 80,
    levels: int = 20,
    max_scatter_points: int = 300,
    dims: Optional[Sequence[int]] = None,
    title: str = "GBS sample density",
):
    """Sample-density view with pairwise marginal support for dim>2."""
    s = np.asarray(samples)
    low = np.asarray(low)
    high = np.asarray(high)
    if s.ndim != 2:
        raise ValueError(f"samples must be [N,D], got {s.shape}")
    d = s.shape[1]
    if low.shape[0] != d or high.shape[0] != d:
        raise ValueError(
            f"low/high shape mismatch: got low={low.shape}, high={high.shape}, D={d}"
        )

    if dims is None:
        dims = tuple(range(d))
    else:
        dims = tuple(dims)
    if len(dims) < 2:
        raise ValueError("Need at least 2 dimensions for pairwise plotting.")

    rng = np.random.default_rng(0)
    n_plot = min(max_scatter_points, s.shape[0])
    idx = rng.choice(s.shape[0], size=n_plot, replace=False)
    s_plot = s[idx]

    # 2D shortcut keeps backward compatibility with existing callers.
    if len(dims) == 2:
        i, j = dims[0], dims[1]
        density, xedges, yedges = np.histogram2d(
            s[:, i],
            s[:, j],
            bins=bins,
            range=[[low[i], high[i]], [low[j], high[j]]],
            density=True,
        )
        xc = 0.5 * (xedges[:-1] + xedges[1:])
        yc = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xc, yc, indexing="xy")

        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111)
        ctf = ax.contourf(X, Y, density.T, levels=levels, cmap="viridis")
        fig.colorbar(ctf, ax=ax, label="estimated density")
        h_scatter = ax.scatter(
            s_plot[:, i], s_plot[:, j], c="r", alpha=0.35, marker="x", label="samples"
        )
        ax.set_xlim(float(low[i]), float(high[i]))
        ax.set_ylim(float(low[j]), float(high[j]))
        ax.set_xlabel(f"dim {i}")
        ax.set_ylabel(f"dim {j}")
        ax.set_title(title)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        fig.subplots_adjust(right=0.80)
        ax.legend(
            handles=[h_scatter],
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )
        return fig, ax

    # dim>2: pairwise marginal-style grid (lower triangle + diagonal histograms).
    k = len(dims)
    fig, axes = plt.subplots(k, k, figsize=(3.5 * k, 3.5 * k), squeeze=False)
    fig.subplots_adjust(wspace=0.15, hspace=0.15)
    mappable = None
    scatter_handle = None

    for row, dim_i in enumerate(dims):
        for col, dim_j in enumerate(dims):
            ax = axes[row, col]

            if row < col:
                ax.axis("off")
                continue

            if row == col:
                ax.hist(s[:, dim_i], bins=30, density=True, alpha=0.8)
                ax.set_xlim(float(low[dim_i]), float(high[dim_i]))
                ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
                if row == k - 1:
                    ax.set_xlabel(f"dim {dim_i}")
                else:
                    ax.set_xticklabels([])
                ax.set_yticklabels([])
                continue

            density, xedges, yedges = np.histogram2d(
                s[:, dim_j],
                s[:, dim_i],
                bins=bins,
                range=[[low[dim_j], high[dim_j]], [low[dim_i], high[dim_i]]],
                density=True,
            )
            xc = 0.5 * (xedges[:-1] + xedges[1:])
            yc = 0.5 * (yedges[:-1] + yedges[1:])
            X, Y = np.meshgrid(xc, yc, indexing="xy")
            ctf = ax.contourf(X, Y, density.T, levels=levels, cmap="viridis")
            mappable = ctf
            scatter_handle = ax.scatter(
                s_plot[:, dim_j],
                s_plot[:, dim_i],
                c="r",
                alpha=0.25,
                s=8,
                marker="x",
                label="samples",
            )
            ax.set_xlim(float(low[dim_j]), float(high[dim_j]))
            ax.set_ylim(float(low[dim_i]), float(high[dim_i]))
            ax.xaxis.set_major_locator(MaxNLocator(nbins=6))
            ax.yaxis.set_major_locator(MaxNLocator(nbins=6))
            if row == k - 1:
                ax.set_xlabel(f"dim {dim_j}")
            else:
                ax.set_xticklabels([])
            if col == 0:
                ax.set_ylabel(f"dim {dim_i}")
            else:
                ax.set_yticklabels([])

    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=axes, fraction=0.02, pad=0.02)
        cbar.set_label("estimated density")
    if scatter_handle is not None:
        fig.legend(
            handles=[scatter_handle],
            loc="center left",
            bbox_to_anchor=(0.92, 0.5),
            frameon=True,
            borderaxespad=0.0,
        )

    fig.suptitle(title, y=0.995)
    fig.tight_layout(rect=[0.0, 0.0, 0.90, 0.98])
    return fig, axes


# -------------------------
# LV loss from precomputed target log-prob VALUES
# -------------------------
def lv_loss_from_values(
    x0, xT, log_ratio, prior_log_prob, target_lp_vals, max_rnd: float | None = None
):
    """
    x0: [B,D]
    xT: [B,D]
    log_ratio: [B]  (or [B,])
    prior_log_prob: callable: prior.log_prob(x0) -> [B]
    target_lp_vals: [B] numeric values (already computed outside JAX)
    """
    running_cost = -log_ratio                      # [B]
    terminal_cost = prior_log_prob(x0) - target_lp_vals  # [B]
    neg_elbo = running_cost + terminal_cost        # [B]
    mask = jnp.isfinite(neg_elbo)
    if max_rnd is not None:
        mask = mask & (neg_elbo < max_rnd)
    neg_elbo_masked = jnp.where(mask, neg_elbo, jnp.nan)
    loss = jnp.nanvar(neg_elbo_masked)
    aux = {
        "train/neg_elbo_mean": jnp.nanmean(neg_elbo_masked),
        "train/neg_elbo_var": jnp.nanvar(neg_elbo_masked),
        "train/running_mean": jnp.nanmean(jnp.where(mask, running_cost, jnp.nan)),
        "train/terminal_mean": jnp.nanmean(jnp.where(mask, terminal_cost, jnp.nan)),
        "train/xT_mean_norm": jnp.mean(jnp.linalg.norm(xT, axis=-1)),
        "train/n_filtered": jnp.sum(~mask),
    }
    return loss, aux


def gbs_trainer(cfg, target, target_log_prob):
    """
    target: used for sampling/eval only if you want; not required for training loss now.
    target_log_prob: function that takes final_state and returns NUMERIC logprob values.
                     IMPORTANT: this is NOT traced by JAX; we call it outside jit.
    """
    key_gen = jax.random.PRNGKey(cfg.seed)
    dim = target.dim
    alg_cfg = cfg.algorithm

    # Prior
    prior = distrax.MultivariateNormalDiag(jnp.zeros(dim), jnp.ones(dim) * alg_cfg.init_std)
    prior_sampler = lambda key: jnp.squeeze(prior.sample(seed=key, sample_shape=(1,)))  # [D]
    prior_log_prob = prior.log_prob  # JAX-traceable

    # Models
    fwd_model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    fwd_params = fwd_model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )
    bwd_model = PISGRADNet(**alg_cfg.model)
    key, key_gen = jax.random.split(key_gen)
    bwd_params = bwd_model.init(
        key,
        jnp.ones([alg_cfg.batch_size, dim]),
        jnp.ones([alg_cfg.batch_size, 1]),
        jnp.ones([alg_cfg.batch_size, dim]),
    )

    optimizer = optax.chain(
        optax.zero_nans(),
        optax.clip(alg_cfg.grad_clip),
        optax.adam(learning_rate=alg_cfg.step_size),
    )
    fwd_state = train_state.TrainState.create(apply_fn=fwd_model.apply, params=fwd_params, tx=optimizer)
    bwd_state = train_state.TrainState.create(apply_fn=bwd_model.apply, params=bwd_params, tx=optimizer)

    # 1) JIT sampler without target
    loss_mode = getattr(alg_cfg, "loss_mode", "tr_lv")  # "dis" or "tr_lv"
    noise_schedule = getattr(
        alg_cfg,
        "noise_schedule",
        VP(
            diff_coeff_sq_min=0.1,
            diff_coeff_sq_max=10.0,
            scale_diff_coeff=1.0,
            terminal_t=1.0,
            generative=False,
            sign=-1.0,
            include_base_drift=True,
        ),
    )
    max_rnd = getattr(alg_cfg, "max_rnd", 1e8)
    raw_sde_ctrl_noise = getattr(alg_cfg, "sde_ctrl_noise", None)
    raw_sde_ctrl_dropout = getattr(alg_cfg, "sde_ctrl_dropout", None)
    sde_ctrl_noise = -1.0 if raw_sde_ctrl_noise is None else float(raw_sde_ctrl_noise)
    sde_ctrl_dropout = -1.0 if raw_sde_ctrl_dropout is None else float(raw_sde_ctrl_dropout)
    process_center = getattr(alg_cfg, "process_center", None)
    if process_center is None and getattr(target, "domain", None) is not None:
        process_center = jnp.asarray(target.domain).mean(axis=-1)
    elif process_center is not None:
        process_center = jnp.asarray(process_center)

    if loss_mode == "dis":
        rnd_jit = jax.jit(
            rnd_no_target,
            static_argnums=(4, 5, 6, 7, 8),  # + stop_grad (must be static for Python branching)
        )
    elif loss_mode == "tr_lv":
        rnd_jit = jax.jit(
            rnd_time_reversal_lv_no_target,
            static_argnums=(3, 4, 5, 6, 7),  # batch_size, prior_sampler, num_steps, process, stop_grad
        )
    else:
        raise ValueError(f"Unknown loss_mode: {loss_mode}")

    # 2) JIT grad of loss
    if loss_mode == "dis":
        def loss_wrapped(
            key,
            model_state,
            fwd_params,
            bwd_params,
            batch_size,
            prior_sampler,
            num_steps,
            noise_schedule,
            target_lp_vals,
        ):
            x0, xT, log_ratio = rnd_no_target(
                key,
                model_state,
                fwd_params,
                bwd_params,
                batch_size,
                prior_sampler,
                num_steps,
                noise_schedule,
                stop_grad=True,
                process_center=process_center,
            )
            loss, aux = lv_loss_from_values(
                x0, xT, log_ratio, prior_log_prob, target_lp_vals, max_rnd=max_rnd
            )
            return loss, aux

        loss_grad = jax.jit(
            jax.grad(loss_wrapped, (2, 3), has_aux=True),
            static_argnums=(4, 5, 6, 7),  # keep callables static
        )
    else:
        def loss_wrapped(
            key,
            model_state,
            fwd_params,
            bwd_params,
            batch_size,
            prior_sampler,
            num_steps,
            process,
            target_lp_vals,
        ):
            x0, xT, rnd_running = rnd_time_reversal_lv_no_target(
                key,
                model_state,
                fwd_params,
                batch_size,
                prior_sampler,
                num_steps,
                process,
                stop_grad=True,
                sde_ctrl_noise=sde_ctrl_noise,
                sde_ctrl_dropout=sde_ctrl_dropout,
                process_center=process_center,
            )
            rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
            loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT, max_rnd=max_rnd)
            return loss, aux

        loss_grad = jax.jit(
            jax.grad(loss_wrapped, (2, 3), has_aux=True),
            static_argnums=(4, 5, 6, 7),
        )

    timer = 0.0
    if loss_mode == "dis":
        history_keys = [
            "train/neg_elbo_mean",
            "train/neg_elbo_var",
            "train/running_mean",
            "train/terminal_mean",
            "train/xT_mean_norm",
            "train/n_filtered",
        ]
    else:
        history_keys = [
            "train/rnd_mean",
            "train/rnd_var",
            "train/xT_mean_norm",
            "train/n_filtered",
        ]
    history = {k: [] for k in history_keys}
    for step in range(alg_cfg.iters):
        key, key_gen = jax.random.split(key_gen)
        iter_time = time()

        model_state = (fwd_state, bwd_state)

        # ---- Phase A: sample xT (JAX) ----
        # We need xT to compute target log-prob values OUTSIDE JAX.
        if loss_mode == "dis":
            x0, xT, log_ratio = rnd_jit(
                key,
                model_state,
                fwd_state.params,
                bwd_state.params,
                alg_cfg.batch_size,
                prior_sampler,
                alg_cfg.num_steps,
                noise_schedule,
                True,  # stop_grad=True
                process_center,
            )
        else:
            x0, xT, rnd_running = rnd_jit(
                key,
                model_state,
                fwd_state.params,
                alg_cfg.batch_size,
                prior_sampler,
                alg_cfg.num_steps,
                noise_schedule,
                True,
                sde_ctrl_noise,
                sde_ctrl_dropout,
                process_center,
            )

        # ---- Phase B: compute target log-prob values outside JAX ----
        # You said you can do: target_log_prob(final_state) once final_state is given.
        # Ensure it returns shape [B], float32/float64.
        target_lp_vals = target_log_prob(xT)  # MUST return a JAX array or something convertible to jnp.array
        target_lp_vals = jnp.asarray(target_lp_vals).reshape(-1)

        # ---- Phase C: compute grads (JAX), using target_lp_vals as input data ----
        (fwd_grads, bwd_grads), aux = loss_grad(
            key,
            model_state,
            fwd_state.params,
            bwd_state.params,
            alg_cfg.batch_size,
            prior_sampler,
            alg_cfg.num_steps,
            noise_schedule,
            target_lp_vals,
        )

        timer += time() - iter_time

        fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
        bwd_state = bwd_state.apply_gradients(grads=bwd_grads)

        for k in history:
            history[k].append(float(aux[k]))

        # Optional: simple logging without full eval framework
        if cfg.use_wandb and (step % getattr(cfg, "log_every", 100) == 0):
            log_dict = {"stats/step": step, "stats/wallclock": timer}
            log_dict.update({k: float(v) for k, v in aux.items()})
            wandb.log(log_dict)

    return {
        "fwd_state": fwd_state,
        "bwd_state": bwd_state,
        "history": history,
        "wallclock_s": timer,
    }
