from __future__ import annotations

import argparse
import functools
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import distrax
import imageio
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from matplotlib.colors import PowerNorm
from matplotlib import pyplot as plt
from tqdm import trange

from learning.module.gbs.gbs_loss import (
    Langevin,
    VP,
    lv_loss_from_rnd,
    rnd_time_reversal_lv_no_target,
)
from learning.module.gbs.gbs_trainer import PISGRADNet


def tanh_box_bijector(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
  """Map unconstrained z to box [low, high] elementwise via tanh."""
  half = 0.5 * (high - low)
  mid = 0.5 * (high + low)
  return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
  """Log|det d x / d z| for x = tanh_box_bijector(z)."""
  z = jnp.atleast_2d(z)
  half = 0.5 * (high - low)
  jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
  return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


def target1_logprob(x: jax.Array) -> jax.Array:
  """Mixture of 2 Gaussians; x: [N,2] or [2]. returns logp: [N]."""
  x = jnp.atleast_2d(x)

  mean1 = jnp.array([1.0, 0.4], dtype=jnp.float32)
  cov1 = 0.3 * jnp.array([[1.0, 0.3], [0.3, 1.0]], dtype=jnp.float32)
  mean2 = jnp.array([-1.0, -0.4], dtype=jnp.float32)
  cov2 = 0.1 * jnp.array([[1.0, -0.3], [-0.3, 1.0]], dtype=jnp.float32)

  d1 = distrax.MultivariateNormalFullCovariance(mean1, cov1)
  d2 = distrax.MultivariateNormalFullCovariance(mean2, cov2)

  logw = jnp.log(jnp.array([0.4, 0.6], dtype=jnp.float32))
  lp = jnp.stack([d1.log_prob(x), d2.log_prob(x)], axis=-1)  # [N, 2]
  return jax.nn.logsumexp(lp + logw, axis=-1)                # [N]


def target2_logprob(z: jax.Array, beta: float = -1.0) -> jax.Array:
  """Energy-based toy target; z: [N,2] or [2]. returns unnormalized logp: [N]."""
  z = jnp.atleast_2d(z)
  z1, z2 = z[:, 0:1], z[:, 1:2]

  r = jnp.hypot(z1, z2)
  logexp1 = -0.5 * jnp.square((z1 - 2.0) / 0.8)
  logexp2 = -0.5 * jnp.square((z1 + 2.0) / 0.8)
  log_mix = jax.nn.logsumexp(
      jnp.concatenate([logexp1, logexp2], axis=-1),
      axis=-1,
      keepdims=True,
  )

  u = 0.5 * jnp.square((r - 4.0) / 0.4) - log_mix
  return (beta * u).squeeze(-1)


def target3_logprob(z: jax.Array, beta: float = -1.0) -> jax.Array:
  """A pointwise (unnormalized) log-density on [0, 1]^2 (2D only).

  This is a corrected version of the "petal/ring" target: it returns a log-density
  value per sample, not a log_softmax over a batch/grid.
  """
  z = jnp.atleast_2d(z)
  x_in, y_in = z[:, 0:1], z[:, 1:2]
  m = 3
  r0 = 0.65
  sr = 0.12
  x = 2.0 * (x_in - 0.5)
  y = 2.0 * (y_in - 0.5)
  r = jnp.hypot(x, y)
  theta = jnp.arctan2(y, x)
  ring = jnp.exp(-0.5 * ((r - r0) / sr) ** 2)
  petals = jnp.cos(m * theta)
  u = jnp.tanh(1.6 * (ring * petals))  # (-1, 1)
  return (-beta * u).squeeze(-1)


def plot_target_contour(
    ax: plt.Axes,
    low: jax.Array,
    high: jax.Array,
    logprob_fn,
    n: int = 200,
    levels: int = 10,
    norm_gamma: float = 0.45,
    title: str = "target",
):
  x, y = jnp.meshgrid(
      jnp.linspace(low[0], high[0], n),
      jnp.linspace(low[1], high[1], n),
      indexing="xy",
  )
  grid = jnp.stack([x.reshape(-1), y.reshape(-1)], axis=-1)
  lp = logprob_fn(grid)
  z = jnp.exp(jnp.clip(lp, a_min=-1000.0)).reshape(n, n)
  ctf = ax.contourf(
      np.array(x),
      np.array(y),
      np.array(z),
      levels=levels,
      cmap="viridis",
      norm=PowerNorm(norm_gamma),
  )
  ax.set_title(title)
  ax.set_xlim(float(low[0]), float(high[0]))
  ax.set_ylim(float(low[1]), float(high[1]))
  ax.set_aspect("equal")
  return ctf


@dataclass(frozen=True)
class RunCfg:
  seed: int = 0
  iters: int = 10000
  batch_size: int = 512
  num_steps: int = 100
  lr: float = 1e-3
  init_std: float = 1.0
  clip_grad: float = 1.0
  sigma_const: float = 1.0
  max_rnd: float = 1e8
  sde_ctrl_noise: float | None = None
  sde_ctrl_dropout: float | None = None

  beta: float = -1.
  gamma: float = 0.45

  save_dir: str = "."
  save_gif: bool = True
  gif_name: str = "GBS_training.gif"
  snap_every: int = 50

  process: str = "vp"  # "vp" or "langevin"
  target: str = "target3"
  use_tanh_bijection: bool = True


def get_target_setup(
    target_name: str, beta: float
) -> tuple[
    Callable[[jax.Array], jax.Array],
    jax.Array,
    jax.Array,
    jax.Array,
    bool,
    jax.Array,
]:
  target_name = target_name.lower()
  if target_name == "target1":
    low = jnp.array([-4.0, -4.0], dtype=jnp.float32)
    high = jnp.array([4.0, 4.0], dtype=jnp.float32)
    prior_loc = jnp.array([0.0, 0.0], dtype=jnp.float32)
    process_center = jnp.array([0.0, 0.0], dtype=jnp.float32)
    return target1_logprob, low, high, prior_loc, False, process_center
  if target_name == "target2":
    low = jnp.array([-6.0, -6.0], dtype=jnp.float32)
    high = jnp.array([6.0, 6.0], dtype=jnp.float32)
    prior_loc = jnp.array([0.0, 0.0], dtype=jnp.float32)
    process_center = jnp.array([0.0, 0.0], dtype=jnp.float32)
    return (
        functools.partial(target2_logprob, beta=beta),
        low,
        high,
        prior_loc,
        False,
        process_center,
    )
  if target_name == "target3":
    low = jnp.array([0.0, 0.0], dtype=jnp.float32)
    high = jnp.array([1.0, 1.0], dtype=jnp.float32)
    prior_loc = jnp.array([0.5, 0.5], dtype=jnp.float32)
    process_center = jnp.array([0.5, 0.5], dtype=jnp.float32)
    return (
        functools.partial(target3_logprob, beta=beta),
        low,
        high,
        prior_loc,
        False,
        process_center,
    )
  raise ValueError(f"Unknown target: {target_name}")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--iters", type=int, default=10000)
  parser.add_argument("--batch_size", type=int, default=512)
  parser.add_argument("--num_steps", type=int, default=100)
  parser.add_argument("--lr", type=float, default=1e-3)
  parser.add_argument("--init_std", type=float, default=1.0)
  parser.add_argument("--clip_grad", type=float, default=1.0)
  parser.add_argument("--sigma_const", type=float, default=1.0)
  parser.add_argument("--max_rnd", type=float, default=1e8)
  parser.add_argument("--sde_ctrl_noise", type=float, default=None)
  parser.add_argument("--sde_ctrl_dropout", type=float, default=None)
  parser.add_argument("--beta", type=float, default=-1.0)
  parser.add_argument("--gamma", type=float, default=0.45)
  parser.add_argument("--save_dir", type=str, default=".")
  parser.add_argument("--save_gif", action="store_true")
  parser.add_argument("--no_save_gif", dest="save_gif", action="store_false")
  parser.set_defaults(save_gif=True)
  parser.add_argument("--gif_name", type=str, default="GBS_training.gif")
  parser.add_argument("--snap_every", type=int, default=50)
  parser.add_argument("--process", choices=["vp", "langevin"], default="vp")
  parser.add_argument(
      "--target",
      choices=["target1", "target2", "target3"],
      default="target3",
  )
  parser.add_argument("--use_tanh_bijection", action="store_true")
  parser.add_argument("--no_use_tanh_bijection", dest="use_tanh_bijection", action="store_false")
  parser.set_defaults(use_tanh_bijection=True)
  args = parser.parse_args()
  cfg = RunCfg(**vars(args))

  os_env_preallocate = "false"
  # Keep this in-script so it works even if user forgets to set env var.
  import os
  os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", os_env_preallocate)

  save_dir = Path(cfg.save_dir)
  save_dir.mkdir(parents=True, exist_ok=True)

  dim = 2
  logprob_fn_x, low, high, prior_loc, clip_prior, process_center = get_target_setup(
      cfg.target, cfg.beta
  )

  if cfg.process == "vp":
    proc = VP(
        diff_coeff_sq_min=0.1,
        diff_coeff_sq_max=10.0,
        scale_diff_coeff=1.0,
        terminal_t=1.0,
        generative=False,
        sign=-1.0,
        include_base_drift=True,
    )
  else:
    proc = Langevin(diff_coeff=cfg.sigma_const, terminal_t=1.0)

  key = jax.random.PRNGKey(cfg.seed)
  sde_ctrl_noise = -1.0 if cfg.sde_ctrl_noise is None else float(cfg.sde_ctrl_noise)
  sde_ctrl_dropout = (
      -1.0 if cfg.sde_ctrl_dropout is None else float(cfg.sde_ctrl_dropout)
  )

  # Optional tanh bijection: train in unconstrained latent space z,
  # evaluate target in bounded x-space.
  if cfg.use_tanh_bijection:
    def to_box(z):
      return tanh_box_bijector(z, low=low, high=high)

    def logprob_fn_train(z):
      x_box = to_box(z)
      return logprob_fn_x(x_box) + tanh_box_logabsdet(z, low=low, high=high)

    latent_prior_loc = jnp.zeros(dim, dtype=jnp.float32)
    process_center = jnp.zeros(dim, dtype=jnp.float32)
  else:
    to_box = lambda z: z
    logprob_fn_train = logprob_fn_x
    latent_prior_loc = prior_loc

  prior = distrax.MultivariateNormalDiag(
      loc=latent_prior_loc,
      scale_diag=jnp.ones(dim, dtype=jnp.float32) * cfg.init_std,
  )
  if clip_prior and (not cfg.use_tanh_bijection):
    prior_sampler = lambda k: jnp.clip(
        jnp.squeeze(prior.sample(seed=k, sample_shape=(1,))), low, high
    )
  else:
    prior_sampler = lambda k: jnp.squeeze(prior.sample(seed=k, sample_shape=(1,)))
  prior_log_prob = prior.log_prob

  model_cfg = dict(dim=dim, num_layers=2, num_hid=64)
  fwd_model = PISGRADNet(**model_cfg)
  bwd_model = PISGRADNet(**model_cfg)  # unused by tr_lv rollout (kept for signature)

  key, k1, k2 = jax.random.split(key, 3)
  fwd_params = fwd_model.init(
      k1,
      jnp.ones([cfg.batch_size, dim]),
      jnp.ones([cfg.batch_size, 1]),
      jnp.ones([cfg.batch_size, dim]),
  )
  bwd_params = bwd_model.init(
      k2,
      jnp.ones([cfg.batch_size, dim]),
      jnp.ones([cfg.batch_size, 1]),
      jnp.ones([cfg.batch_size, dim]),
  )

  opt = optax.chain(
      optax.zero_nans(),
      optax.clip(cfg.clip_grad),
      optax.adam(cfg.lr),
  )
  fwd_state = train_state.TrainState.create(
      apply_fn=fwd_model.apply, params=fwd_params, tx=opt
  )
  bwd_state = train_state.TrainState.create(
      apply_fn=bwd_model.apply, params=bwd_params, tx=opt
  )

  sampler_jit = jax.jit(
      rnd_time_reversal_lv_no_target,
      static_argnums=(3, 4, 5, 6, 7),
  )

  def loss_wrapped(key, model_state, fwd_params, bwd_params):
    x0, xT, rnd_running = rnd_time_reversal_lv_no_target(
        key,
        model_state,
        fwd_params,
        cfg.batch_size,
        prior_sampler,
        cfg.num_steps,
        proc,
        stop_grad=True,
        sde_ctrl_noise=sde_ctrl_noise,
        sde_ctrl_dropout=sde_ctrl_dropout,
        process_center=process_center,
    )
    target_lp_vals = jnp.asarray(logprob_fn_train(xT)).reshape(-1)
    rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
    loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT, max_rnd=cfg.max_rnd)
    return loss, aux

  loss_grad = jax.jit(jax.grad(loss_wrapped, (2, 3), has_aux=True))

  hist = {
      k: []
      for k in ["train/rnd_mean", "train/rnd_var", "train/xT_mean_norm", "train/n_filtered"]
  }
  frames: list[np.ndarray] = []
  last_xT = None

  for t in trange(cfg.iters):
    key, k_step = jax.random.split(key)
    model_state = (fwd_state, bwd_state)

    # sample (for visualization)
    _, xT, _ = sampler_jit(
        k_step,
        model_state,
        fwd_state.params,
        cfg.batch_size,
        prior_sampler,
        cfg.num_steps,
        proc,
        True,
        sde_ctrl_noise,
        sde_ctrl_dropout,
        process_center,
    )
    last_xT = xT

    (fwd_grads, bwd_grads), aux = loss_grad(
        k_step, model_state, fwd_state.params, bwd_state.params
    )
    fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
    bwd_state = bwd_state.apply_gradients(grads=bwd_grads)

    for k in hist:
      hist[k].append(float(aux[k]))

    if t % max(cfg.snap_every, 1) == 0:
      print(t, {k: hist[k][-1] for k in hist})

    if cfg.save_gif and (t % max(cfg.snap_every, 1) == 0):
      fig, ax = plt.subplots(1, 1, figsize=(5, 5))
      ctf = plot_target_contour(
          ax,
          low,
          high,
          logprob_fn_x,
          n=200,
          levels=10,
          norm_gamma=cfg.gamma,
          title=f"GBS xT (iter {t}) {cfg.target} beta={cfg.beta}",
      )
      fig.colorbar(ctf, ax=ax)
      pts = np.array(to_box(last_xT))
      ax.scatter(pts[:, 0], pts[:, 1], s=3, alpha=0.25, c="r")
      fig.tight_layout()
      fig.canvas.draw()
      frame = np.asarray(fig.canvas.buffer_rgba())[..., :3]
      frames.append(frame)
      plt.close(fig)

  if cfg.save_gif and frames:
    gif_path = save_dir / cfg.gif_name
    imageio.mimsave(gif_path.as_posix(), frames, fps=4)
    print("Saved GIF to:", gif_path)

  # final big sample for saving/plotting
  key, k_final = jax.random.split(key)
  b = 2**14
  x0, xT_final, rnd_running = rnd_time_reversal_lv_no_target(
      k_final,
      (fwd_state, bwd_state),
      fwd_state.params,
      b,
      prior_sampler,
      cfg.num_steps,
      proc,
      stop_grad=True,
      sde_ctrl_noise=sde_ctrl_noise,
      sde_ctrl_dropout=sde_ctrl_dropout,
      process_center=process_center,
  )
  xT_final_box = to_box(xT_final)
  np.save((save_dir / "gbs_samples.npy").as_posix(), np.array(xT_final_box))

  fig, ax = plt.subplots(1, 1, figsize=(5, 5))
  ctf = plot_target_contour(
      ax,
      low,
      high,
      logprob_fn_x,
      n=200,
      levels=10,
      norm_gamma=cfg.gamma,
      title="Target density + final xT",
  )
  fig.colorbar(ctf, ax=ax)
  pts = np.array(xT_final_box)
  ax.scatter(pts[:, 0], pts[:, 1], s=1, alpha=0.25, c="r")
  fig.tight_layout()
  fig.savefig((save_dir / "gbs_final.png").as_posix(), dpi=150)
  plt.show()


if __name__ == "__main__":
  main()
