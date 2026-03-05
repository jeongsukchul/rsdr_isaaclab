from .gmmvi.network import GMMTrainingState, create_gmm_network_and_state
from .sampler import LearnableSampler
import jax
import jax.numpy as jnp
import torch
import logging

# Silence JAX debug logs
logging.getLogger("jax").setLevel(logging.WARNING)
logging.getLogger("jax._src").setLevel(logging.WARNING)

# (optional) if you configured root logging to DEBUG somewhere:
logging.basicConfig(level=logging.INFO)  # or WARNING
def jax_to_torch(x):
    """JAX array -> Torch tensor (zero-copy via DLPack when possible)."""
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch_to_jax(t: torch.Tensor):
    """Torch tensor -> JAX array (zero-copy via DLPack when possible)."""
    # Ensure contiguous to avoid surprises
    if not t.is_contiguous():
        t = t.contiguous()
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))


def _almost_uniform_mapping(num_samples: int, num_components: int, key: jax.Array) -> jax.Array:
    """Near-uniform component ids (counts differ by at most 1), then shuffled."""
    num_components = max(1, int(num_components))
    base = jnp.arange(num_samples, dtype=jnp.int32) % jnp.int32(num_components)
    return jax.random.permutation(key, base)
class GMMVI(LearnableSampler):
    def __init__(self, cfg, device: str, 
                 beta=1.0, 
                 num_envs=128,
                 batch_size=1024, 
                 **kwargs):
        super().__init__(cfg, device)
        self.name = "GMMVI"
        bound_info = (torch_to_jax(self.low), torch_to_jax(self.high))
        rng = jax.random.PRNGKey(torch.cuda.initial_seed() % (2**32))
        self.rng, init_key = jax.random.split(rng) 
        init_gmmvi_state, gmm_network = create_gmm_network_and_state(cfg.total_params, \
                                                               num_envs, batch_size, init_key,\
                                                                bound_info=bound_info)
        self.gmmvi_state = init_gmmvi_state
        self.gmm_network = gmm_network
        self.update_fn = jax.jit(make_gmm_update(gmm_network))
        self.sample_size = batch_size
        self.beta = float(beta)
        self.current_dist= None
    def get_train_sample_fn(self):
        return lambda num_samples: self.sample_model(num_samples)

    def sample_model(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.rng, key = jax.random.split(self.rng)
        samples_jax, mapping_jax = self.gmm_network.model.sample(
            self.gmmvi_state.model_state.gmm_state, key, num_samples
        )
        samples_torch = jax_to_torch(samples_jax).to(device=self.device, dtype=torch.float32)
        mapping_torch = jax_to_torch(mapping_jax).to(device=self.device, dtype=torch.int32)
        return samples_torch, mapping_torch

    def sample_training(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        self.rng, key = jax.random.split(self.rng)
        samples_jax, mapping_jax = self.gmm_network.sample_selector.select_samples(
            self.gmmvi_state.model_state,
            key,
        )
        n_sel = int(samples_jax.shape[0])
        if n_sel > num_samples:
            samples_jax = samples_jax[:num_samples]
            mapping_jax = mapping_jax[:num_samples]
        elif n_sel < num_samples:
            deficit = num_samples - n_sel
            self.rng, key_pad = jax.random.split(self.rng)
            pad_samples, pad_mapping = self.gmm_network.model.sample(
                self.gmmvi_state.model_state.gmm_state, key_pad, deficit
            )
            samples_jax = jnp.concatenate([samples_jax, pad_samples], axis=0)
            mapping_jax = jnp.concatenate([mapping_jax, pad_mapping.astype(jnp.int32)], axis=0)
        samples_torch = jax_to_torch(samples_jax).to(device=self.device, dtype=torch.float32)
        mapping_torch = jax_to_torch(mapping_jax).to(device=self.device, dtype=torch.int32)
        return samples_torch, mapping_torch

    def sample(self, num_samples: int) -> torch.Tensor:
        return self.sample_model(num_samples)

    def sample_contexts(self, num_samples: int) -> torch.Tensor:
        samples, _ = self.sample_model(num_samples)
        return samples
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value_jax = torch_to_jax(value.to("cuda" if value.is_cuda else "cpu"))
        lp_jax = self.gmm_network.model.log_density(self.gmmvi_state.model_state.gmm_state, value_jax)
        return jax_to_torch(lp_jax).to(device=value.device)
    def log_prob_batch(self, values: torch.Tensor) -> torch.Tensor:
        """Batched log-density helper for diagnostics."""
        values = values.to(device=self.device, dtype=torch.float32)
        values_jax = torch_to_jax(values)
        lp_jax = jax.vmap(self.gmm_network.model.log_density, in_axes=(None, 0))(
            self.gmmvi_state.model_state.gmm_state, values_jax
        )
        return jax_to_torch(lp_jax).to(device=values.device, dtype=torch.float32)
    def update(self, samples_torch, mapping_torch, returns):

        returns = returns.reshape(-1).to(device=samples_torch.device, dtype=torch.float32)
        mapping_torch = mapping_torch.reshape(-1).to(device=samples_torch.device, dtype=torch.int32)

        returns_jax = torch_to_jax(returns)
        samples_jax = torch_to_jax(samples_torch)
        mapping_jax = torch_to_jax(mapping_torch).astype(jnp.int32)
        target_lnpdfs = returns_jax * self.beta

        # Initial bootstrap only:
        # when batch_size > num_envs, fill the first DB batch to sample_size
        # using zero-target samples and (almost) uniform mapping.
        written_before = int(jax.device_get(self.gmmvi_state.sample_db_state.num_samples_written)[0])
        if written_before < self.sample_size:
            deficit = max(0, int(self.sample_size - samples_jax.shape[0]))
            if deficit > 0:
                num_components = int(self.gmmvi_state.model_state.gmm_state.num_components)
                self.rng, map_key, sample_key = jax.random.split(self.rng, 3)
                pad_mapping = _almost_uniform_mapping(deficit, num_components, map_key)
                pad_samples, _ = self.gmm_network.model.sample_from_components_shuffle(
                    self.gmmvi_state.model_state.gmm_state,
                    pad_mapping,
                    sample_key,
                )
                samples_jax = jnp.concatenate([samples_jax, pad_samples], axis=0)
                mapping_jax = jnp.concatenate([mapping_jax, pad_mapping], axis=0)
                target_lnpdfs = jnp.concatenate(
                    [target_lnpdfs, jnp.zeros((deficit,), dtype=target_lnpdfs.dtype)],
                    axis=0,
                )

        self.rng, update_key = jax.random.split(self.rng)
        new_sample_db_state = self.gmm_network.sample_selector.save_samples(
            self.gmmvi_state.model_state,
            self.gmmvi_state.sample_db_state,
            samples_jax,                 # or maybe "scores" depending on your API
            target_lnpdfs,
            jnp.zeros_like(samples_jax),
            mapping_jax
        )

        new_gmm_training_state = self.gmmvi_state._replace(sample_db_state=new_sample_db_state)
        self.gmmvi_state = self.update_fn(new_gmm_training_state, update_key)


def make_gmm_update(gmm_network):
  def gmm_update(
          gmmvi_state, key
    ):
      samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads = \
          gmm_network.sample_selector.select_train_datas(gmmvi_state.sample_db_state)
      new_component_stepsizes = gmm_network.component_stepsize_fn(gmmvi_state.model_state)
      new_model_state = gmm_network.model.update_stepsizes(gmmvi_state.model_state, new_component_stepsizes)
      expected_hessian_neg, expected_grad_neg = gmm_network.more_ng_estimator(new_model_state,
                                                              samples,
                                                              sample_dist_densities,
                                                              target_lnpdfs,
                                                              target_lnpdf_grads)
      new_model_state = gmm_network.component_updater(new_model_state,
                                      expected_hessian_neg,
                                      expected_grad_neg,
                                      new_model_state.stepsizes)
          
      new_model_state = gmm_network.weight_updater(new_model_state, samples, sample_dist_densities, target_lnpdfs,
                                                      gmmvi_state.weight_stepsize)
      new_num_updates = gmmvi_state.num_updates + 1
      # new_model_state, new_component_adapter_state, new_sample_db_state = \
          # gmm_network.component_adapter(gmmvi_state.component_adaptation_state,
          #                                             gmmvi_state.sample_db_state,
          #                                             new_model_state,
          #                                             new_num_updates,
                                                      # key)

      return GMMTrainingState(temperature=gmmvi_state.temperature,
                          model_state=new_model_state,
                          component_adaptation_state=gmmvi_state.component_adaptation_state,#new_component_adapter_state,
                          num_updates=new_num_updates,
                          sample_db_state=gmmvi_state.sample_db_state,#new_sample_db_state,
                          weight_stepsize=gmmvi_state.weight_stepsize)
  return gmm_update
