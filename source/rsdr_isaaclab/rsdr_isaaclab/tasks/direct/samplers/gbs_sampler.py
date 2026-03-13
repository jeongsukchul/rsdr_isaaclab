from __future__ import annotations

import logging

import distrax
import jax
import jax.numpy as jnp
import optax
import torch
from flax.training import train_state as flax_train_state

from .gbs.gbs_loss import Langevin, VP, lv_loss_from_rnd, rnd_time_reversal_lv_no_target
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


def gbs_sample_log_prob(
    x0: jax.Array,
    rnd_running: jax.Array,
    prior_log_prob,
    logabsdet: jax.Array | None = None,
) -> jax.Array:
    log_prob = prior_log_prob(x0) + rnd_running
    if logabsdet is not None:
        log_prob = log_prob - logabsdet
    return log_prob


class GBS(LearnableSampler):
    def __init__(
        self,
        cfg,
        device: str,
        beta: float = 1.0,
        batch_size: int = 1024,
        init_std: float = 1.0,
        lr: float = 1e-4,
        clip_grad: float = 1.0,
        num_steps: int = 32,
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
        self._last_sample_contexts: torch.Tensor | None = None
        self._last_sample_log_probs: torch.Tensor | None = None

        self.low_jax = torch_to_jax(self.low)
        self.high_jax = torch_to_jax(self.high)
        self.dim = int(self.num_params)

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
        batch_size = int(batch_size)
        if batch_size in self._train_step_fns:
            return self._train_step_fns[batch_size]

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
        def train_step_jit(key, fwd_state, bwd_state, target_lnpdf):
            def loss_from_params(fwd_params, bwd_params):
                x0, xT, rnd_running = rnd_time_reversal_lv_no_target(
                    key,
                    (fwd_state, bwd_state),
                    fwd_params,
                    batch_size,
                    prior_sampler,
                    num_steps,
                    process,
                    True,
                    sde_ctrl_noise,
                    sde_ctrl_dropout,
                    process_center,
                )
                target_lp_vals = target_lnpdf
                if use_tanh_bijection:
                    target_lp_vals = target_lp_vals + tanh_box_logabsdet(xT, low, high)
                rnd_total = prior_log_prob(x0) + rnd_running - target_lp_vals
                loss, aux, _ = lv_loss_from_rnd(rnd_total, xT=xT, max_rnd=max_rnd)
                return loss, aux

            (grads, aux) = jax.grad(loss_from_params, (0, 1), has_aux=True)(
                fwd_state.params, bwd_state.params
            )
            fwd_grads, bwd_grads = grads
            new_fwd_state = fwd_state.apply_gradients(grads=fwd_grads)
            new_bwd_state = bwd_state.apply_gradients(grads=bwd_grads)
            return new_fwd_state, new_bwd_state, aux

        self._train_step_fns[batch_size] = train_step_jit
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
        return x0, latent_samples, rnd_running

    def sample_model(self, num_samples: int) -> torch.Tensor:
        x0, latent_samples, rnd_running = self._latent_samples(num_samples)
        samples = self._latent_to_box(latent_samples)
        logabsdet = (
            tanh_box_logabsdet(latent_samples, self.low_jax, self.high_jax)
            if self.use_tanh_bijection
            else None
        )
        log_probs = gbs_sample_log_prob(
            x0=x0,
            rnd_running=rnd_running,
            prior_log_prob=self.prior.log_prob,
            logabsdet=logabsdet,
        )
        samples_torch = jax_to_torch(samples).to(device=self.device, dtype=torch.float32)
        log_probs_torch = jax_to_torch(log_probs).to(device=self.device, dtype=torch.float32)
        self._last_sample_contexts = samples_torch.detach().clone()
        self._last_sample_log_probs = log_probs_torch.detach().clone()
        return samples_torch

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.sample_model(num_samples)

    def sample_contexts(self, num_samples: int) -> torch.Tensor:
        return self.sample_model(num_samples)

    def get_train_sample_fn(self):
        return lambda num_samples: self.sample_model(num_samples)

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value = value.to(device=self.device, dtype=torch.float32)
        batch_value = value.unsqueeze(0) if value.ndim == 1 else value
        if (
            self._last_sample_contexts is not None
            and self._last_sample_log_probs is not None
            and batch_value.shape == self._last_sample_contexts.shape
            and torch.allclose(batch_value, self._last_sample_contexts, atol=1e-6, rtol=1e-5)
        ):
            result = self._last_sample_log_probs.to(device=value.device, dtype=value.dtype)
            return result[0] if value.ndim == 1 else result

        batch = batch_value.shape[0]
        return torch.full((batch,), float("nan"), device=value.device, dtype=value.dtype)

    def log_prob_batch(self, values: torch.Tensor) -> torch.Tensor:
        return self.log_prob(values)

    def update(self, contexts, returns):
        del contexts
        returns = returns.reshape(-1).to(device=self.device, dtype=torch.float32)
        target_lnpdf = torch_to_jax(returns) * self.beta
        batch_size = int(target_lnpdf.shape[0])
        train_step_jit = self._get_train_step_jit(batch_size)

        aux = None
        for _ in range(self.train_steps_per_update):
            self.rng, sample_key = jax.random.split(self.rng)
            self.fwd_state, self.bwd_state, aux = train_step_jit(
                sample_key,
                self.fwd_state,
                self.bwd_state,
                target_lnpdf,
            )

        if aux is not None:
            self.last_aux = {k: float(v) for k, v in jax.device_get(aux).items()}
