from __future__ import annotations

import logging
from collections import deque

import distrax
import jax
import jax.numpy as jnp
import optax
import torch
from flax.training import train_state as flax_train_state

from .gbs.gbs_loss import (
    Langevin,
    VP,
    lv_loss_from_rnd,
    rnd_time_reversal_lv_from_seeds,
    rnd_time_reversal_lv_no_target,
)
from .gbs.gbs_trainer import PISGRADNet
from .sampler import LearnableSampler

# Keep JAX logging quiet in the training loop.
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)


def jax_to_torch(x):
    try:
        return torch.utils.dlpack.from_dlpack(x)
    except TypeError:
        return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


def torch_to_jax(t: torch.Tensor):
    if not t.is_contiguous():
        t = t.contiguous()
    try:
        return jax.dlpack.from_dlpack(t)
    except TypeError:
        return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))


def tanh_box_bijector(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    half = 0.5 * (high - low)
    mid = 0.5 * (high + low)
    return mid + half * jnp.tanh(z)


def tanh_box_logabsdet(z: jax.Array, low: jax.Array, high: jax.Array) -> jax.Array:
    z = jnp.atleast_2d(z)
    half = 0.5 * (high - low)
    jac_diag = half * (1.0 - jnp.tanh(z) ** 2)
    return jnp.sum(jnp.log(jnp.clip(jac_diag, 1e-12)), axis=-1)


class GBS(LearnableSampler):
    def __init__(
        self,
        cfg,
        device: str,
        beta: float = 1.0,
        batch_size: int = 1024,
        init_std: float = 1.0,
        lr: float = 1e-3,
        clip_grad: float = 1.0,
        num_steps: int = 100,
        model_num_layers: int = 2,
        model_num_hid: int = 64,
        max_rnd: float = 1e8,
        sde_ctrl_noise: float | None = None,
        sde_ctrl_dropout: float | None = None,
        use_tanh_bijection: bool = True,
        clip_prior_to_bounds: bool = False,
        process_type: str = "vp",
        diff_coeff_sq_min: float = 0.1,
        diff_coeff_sq_max: float = 10.0,
        scale_diff_coeff: float = 1.0,
        sigma_const: float = 1.0,
        terminal_t: float = 1.0,
        train_steps_per_update: int = 1,
        process_center=None,
        **kwargs,
    ):
        super().__init__(cfg, device)
        self.name = "GBS"
        self.dist_dim = self.learning_num_params
        self.beta = float(beta)
        self.batch_size = int(batch_size)
        self.init_std = float(init_std)
        self.lr = float(lr)
        self.clip_grad = float(clip_grad)
        self.num_steps = int(num_steps)
        self.model_num_layers = int(model_num_layers)
        self.model_num_hid = int(model_num_hid)
        self.max_rnd = float(max_rnd)
        self.sde_ctrl_noise = -1.0 if sde_ctrl_noise is None else float(sde_ctrl_noise)
        self.sde_ctrl_dropout = -1.0 if sde_ctrl_dropout is None else float(sde_ctrl_dropout)
        self.use_tanh_bijection = bool(use_tanh_bijection)
        self.clip_prior_to_bounds = bool(clip_prior_to_bounds)
        self.train_steps_per_update = max(1, int(train_steps_per_update))
        self.current_dist = None
        self.last_aux: dict[str, float] = {}
        self._pending_sample_seeds: jax.Array | None = None
        self._pending_sample_contexts: deque[torch.Tensor] = deque()

        self.low_jax = torch_to_jax(self.learn_low)
        self.high_jax = torch_to_jax(self.learn_high)
        self.dim = int(self.learning_num_params)

        if not self.has_learning_dims:
            self.current_dist = None
            self.prior = None
            self.prior_sampler = None
            self.process = None
            self.rng = None
            self.model_fwd = None
            self.model_bwd = None
            self.fwd_state = None
            self.bwd_state = None
            self._sampler_fns: dict[int, callable] = {}
            self._train_step_fns: dict[int, callable] = {}
            return

        if self.use_tanh_bijection:
            self._latent_to_box = lambda z: tanh_box_bijector(z, self.low_jax, self.high_jax)
            latent_prior_loc = jnp.zeros(self.dim, dtype=jnp.float32)
            self.process_center = jnp.zeros(self.dim, dtype=jnp.float32)
        else:
            self._latent_to_box = lambda z: z
            latent_prior_loc = 0.5 * (self.low_jax + self.high_jax)
            if process_center is None:
                self.process_center = latent_prior_loc
            else:
                self.process_center = jnp.asarray(process_center, dtype=jnp.float32)

        self.prior = distrax.MultivariateNormalDiag(
            loc=latent_prior_loc,
            scale_diag=jnp.ones(self.dim, dtype=jnp.float32) * self.init_std,
        )
        if self.clip_prior_to_bounds and (not self.use_tanh_bijection):
            self.prior_sampler = lambda key: jnp.clip(
                jnp.squeeze(self.prior.sample(seed=key, sample_shape=(1,))),
                self.low_jax,
                self.high_jax,
            )
        else:
            self.prior_sampler = lambda key: jnp.squeeze(
                self.prior.sample(seed=key, sample_shape=(1,))
            )

        if process_type.lower() == "langevin":
            self.process = Langevin(diff_coeff=float(sigma_const), terminal_t=float(terminal_t))
        else:
            self.process = VP(
                diff_coeff_sq_min=float(diff_coeff_sq_min),
                diff_coeff_sq_max=float(diff_coeff_sq_max),
                scale_diff_coeff=float(scale_diff_coeff),
                terminal_t=float(terminal_t),
                generative=False,
                sign=-1.0,
                include_base_drift=True,
            )

        seed = torch.cuda.initial_seed() if torch.cuda.is_available() else torch.initial_seed()
        rng = jax.random.PRNGKey(seed % (2**32))
        self.rng, gbs_key_fwd, gbs_key_bwd = jax.random.split(rng, 3)
        self.model_fwd = PISGRADNet(
            dim=self.dim,
            num_layers=self.model_num_layers,
            num_hid=self.model_num_hid,
        )
        self.model_bwd = PISGRADNet(
            dim=self.dim,
            num_layers=self.model_num_layers,
            num_hid=self.model_num_hid,
        )

        init_x = jnp.ones([self.batch_size, self.dim], dtype=jnp.float32)
        init_t = jnp.ones([self.batch_size, 1], dtype=jnp.float32)
        init_lgv = jnp.ones([self.batch_size, self.dim], dtype=jnp.float32)
        gbs_fwd_params = self.model_fwd.init(gbs_key_fwd, init_x, init_t, init_lgv)
        gbs_bwd_params = self.model_bwd.init(gbs_key_bwd, init_x, init_t, init_lgv)

        gbs_optimizer = optax.chain(
            optax.zero_nans(),
            optax.clip(self.clip_grad),
            optax.adam(learning_rate=self.lr),
        )
        self.fwd_state = flax_train_state.TrainState.create(
            apply_fn=self.model_fwd.apply, params=gbs_fwd_params, tx=gbs_optimizer
        )
        self.bwd_state = flax_train_state.TrainState.create(
            apply_fn=self.model_bwd.apply, params=gbs_bwd_params, tx=gbs_optimizer
        )

        self._sampler_fns: dict[int, callable] = {}
        self._train_step_fns: dict[int, callable] = {}

    @property
    def flow_state(self):
        return self.fwd_state, self.bwd_state

    def _get_sampler_jit(self, batch_size: int):
        batch_size = int(batch_size)
        if batch_size not in self._sampler_fns:
            self._sampler_fns[batch_size] = jax.jit(
                rnd_time_reversal_lv_no_target,
                static_argnums=(3, 4, 5, 6, 7),
            )
        return self._sampler_fns[batch_size]

    def _get_train_step_jit(self, batch_size: int):

        prior_log_prob = self.prior.log_prob
        prior_sampler = self.prior_sampler
        process = self.process
        max_rnd = self.max_rnd
        use_tanh_bijection = self.use_tanh_bijection
        low = self.low_jax
        high = self.high_jax
        num_steps = self.num_steps
        sde_ctrl_noise = self.sde_ctrl_noise
        sde_ctrl_dropout = self.sde_ctrl_dropout
        process_center = self.process_center

        @jax.jit
        def train_step_jit(sample_seeds, fwd_state, bwd_state, target_lnpdf):
            def loss_from_params(fwd_params, bwd_params):
                x0, xT, rnd_running = rnd_time_reversal_lv_from_seeds(
                    sample_seeds,
                    (fwd_state, bwd_state),
                    fwd_params,
                    prior_sampler,
                    num_steps,
                    process,
                    True,
                    sde_ctrl_noise,
                    sde_ctrl_dropout,
                    process_center,
                )
                target_lp_vals = target_lnpdf
                logabsdet = None
                if use_tanh_bijection:
                    target_lp_vals = target_lp_vals + tanh_box_logabsdet(xT, low, high)
                rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
                xT_box = self._latent_to_box(xT)
                loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT_box, max_rnd=max_rnd)
                return loss, aux

            (grads, aux) = jax.grad(loss_from_params, (0, 1), has_aux=True)(
                fwd_state.params, bwd_state.params
            )
            fwd_grads, bwd_grads = grads
            new_fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
            new_bwd_state = bwd_state.apply_gradients(grads=bwd_grads)
            return new_fwd_state, new_bwd_state, aux

        return train_step_jit

    def _latent_samples(self, num_samples: int):
        self.rng, sample_key = jax.random.split(self.rng)
        sampler_jit = self._get_sampler_jit(num_samples)
        x0, latent_samples, rnd_running = sampler_jit(
            sample_key,
            self.flow_state,
            self.fwd_state.params,
            int(num_samples),
            self.prior_sampler,
            self.num_steps,
            self.process,
            True,
            self.sde_ctrl_noise,
            self.sde_ctrl_dropout,
            self.process_center,
        )
        sample_seeds = jax.random.split(sample_key, int(num_samples))
        return x0, latent_samples, rnd_running, sample_seeds

    def _append_pending_sample_metadata(self, sample_seeds: jax.Array, samples_torch: torch.Tensor):
        if self._pending_sample_seeds is None:
            self._pending_sample_seeds = sample_seeds
        else:
            self._pending_sample_seeds = jnp.concatenate([self._pending_sample_seeds, sample_seeds], axis=0)
        self._pending_sample_contexts.append(samples_torch.detach().clone())

    def _pop_pending_sample_metadata(self, batch_size: int) -> tuple[jax.Array, torch.Tensor]:
        if self._pending_sample_seeds is None or int(self._pending_sample_seeds.shape[0]) < batch_size:
            pending = 0 if self._pending_sample_seeds is None else int(self._pending_sample_seeds.shape[0])
            raise RuntimeError(
                f"GBS update requested {batch_size} trajectories, but only {pending} are buffered."
            )

        sample_seeds = self._pending_sample_seeds[:batch_size]
        self._pending_sample_seeds = self._pending_sample_seeds[batch_size:]
        if int(self._pending_sample_seeds.shape[0]) == 0:
            self._pending_sample_seeds = None

        context_chunks: list[torch.Tensor] = []
        remaining = batch_size
        while remaining > 0:
            if not self._pending_sample_contexts:
                raise RuntimeError("GBS sample context buffer underflow while building an update batch.")
            chunk = self._pending_sample_contexts.popleft()
            take = min(remaining, int(chunk.shape[0]))
            context_chunks.append(chunk[:take])
            if take < int(chunk.shape[0]):
                self._pending_sample_contexts.appendleft(chunk[take:])
            remaining -= take
        return sample_seeds, torch.cat(context_chunks, dim=0)

    def sample_model(self, num_samples: int, record_for_update: bool = False) -> torch.Tensor:
        if not self.has_learning_dims:
            return self._assemble_full_contexts(None, num_samples)
        _, latent_samples, _, sample_seeds = self._latent_samples(num_samples)
        samples = self._latent_to_box(latent_samples)
        learned_samples_torch = jax_to_torch(samples).to(device=self.device, dtype=torch.float32)
        samples_torch = self._assemble_full_contexts(learned_samples_torch, num_samples)
        if record_for_update:
            self._append_pending_sample_metadata(sample_seeds, samples_torch)

        return samples_torch

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.sample_model(num_samples)

    def sample_contexts(self, num_samples: int) -> torch.Tensor:
        return self.sample_model(num_samples)

    def get_train_sample_fn(self):
        return lambda num_samples: self.sample_model(num_samples, record_for_update=True)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        if not self.has_learning_dims:
            batch = value.shape[0] if value.ndim > 1 else 1
            return torch.zeros((batch,), device=value.device, dtype=value.dtype)
        value = value.to(device=self.device, dtype=torch.float32)
        batch_value = value.unsqueeze(0) if value.ndim == 1 else value
        batch = batch_value.shape[0]
        return torch.full((batch,), float("nan"), device=value.device, dtype=value.dtype)

    def log_prob_batch(self, values: torch.Tensor) -> torch.Tensor:
        return self.log_prob(values)

    def update(self, contexts, returns):
        if not self.has_learning_dims:
            return
        contexts = contexts.to(device=self.device, dtype=torch.float32)
        returns = returns.reshape(-1).to(device=self.device, dtype=torch.float32)
        if contexts.ndim != 2 or contexts.shape[0] != returns.shape[0]:
            raise ValueError(
                f"GBS.update expected contexts [B,D] aligned with returns [B], got "
                f"contexts={tuple(contexts.shape)} returns={tuple(returns.shape)}"
            )
        target_lnpdf = torch_to_jax(returns) * self.beta
        batch_size = int(target_lnpdf.shape[0])
        train_step_jit = self._get_train_step_jit(batch_size)
        sample_seeds, buffered_contexts = self._pop_pending_sample_metadata(batch_size)

        contexts = self._project_learning_contexts(contexts)
        buffered_contexts = self._project_learning_contexts(buffered_contexts)

        context_error = (buffered_contexts.to(self.device) - contexts).abs().max().item()

        for _ in range(self.train_steps_per_update):
            self.fwd_state, self.bwd_state, aux = train_step_jit(
                sample_seeds,
                self.fwd_state,
                self.bwd_state,
                target_lnpdf,
            )
        self.last_aux = {k: float(v) for k, v in aux.items()}
        self.last_aux["train/replay_context_max_abs_err"] = float(context_error)
