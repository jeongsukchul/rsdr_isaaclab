from .gmmvi.network import GMMTrainingState, create_gmm_network_and_state
from .sampler import LearnableSampler
import jax
import jax.numpy as jnp
import torch
def jax_to_torch(x):
    """JAX array -> Torch tensor (zero-copy via DLPack when possible)."""
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))

def torch_to_jax(t: torch.Tensor):
    """Torch tensor -> JAX array (zero-copy via DLPack when possible)."""
    # Ensure contiguous to avoid surprises
    if not t.is_contiguous():
        t = t.contiguous()
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(t))
class GMMVI(LearnableSampler):
    def __init__(self, cfg, device: str, 
                 beta=1.0, 
                 batch_size=128, 
                 **kwargs):
        super().__init__(cfg, device)
        self.name = "GMMVI"
        bound_info = (self.low.tolist(), self.high.tolist())
        rng = jax.random.PRNGKey(torch.cuda.initial_seed() % (2**32))
        self.rng, init_key = jax.random.split(rng) 
        init_gmmvi_state, gmm_network = create_gmm_network_and_state(cfg.total_params, \
                                                               batch_size, batch_size, init_key,\
                                                               prior_scale=.1,
                                                                bound_info=bound_info)
        self.gmmvi_state = init_gmmvi_state
        self.gmm_network = gmm_network
        self.update_fn = jax.jit(make_gmm_update(gmm_network))
        self.sample_size = batch_size
        self.beta = beta
        self.current_dist= None
    def get_train_sample_fn(self):
        return lambda num_samples: self.sample(num_samples)
    def sample(self, num_samples: int) -> torch.Tensor:
        self.rng, key = jax.random.split(self.rng)
        samples_jax, mapping_jax = self.gmm_network.model.sample(
            self.gmmvi_state.model_state.gmm_state, key, num_samples
        )

        samples_torch = jax_to_torch(samples_jax).to(device=self.device, dtype=torch.float32)
        # mapping might be int/bool; keep dtype
        mapping_torch = jax_to_torch(mapping_jax).to(device=self.device)
        return samples_torch, mapping_torch
    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        value_jax = torch_to_jax(value.to("cuda" if value.is_cuda else "cpu"))
        lp_jax = self.gmm_network.model.log_density(self.gmmvi_state.model_state.gmm_state, value_jax)
        return jax_to_torch(lp_jax).to(device=value.device)
    def update(self, contexts, returns):
        samples_torch, mapping_torch = contexts

        returns_jax  = torch_to_jax(returns)
        samples_jax  = torch_to_jax(samples_torch)
        mapping_jax  = torch_to_jax(mapping_torch)

        target_lnpdfs = -returns_jax * self.beta
        self.rng, update_key = jax.random.split(self.rng)

        new_sample_db_state = self.gmm_network.sample_selector.save_samples(
            self.gmmvi_state.model_state,
            self.gmmvi_state.sample_db_state,
            returns_jax,                 # or maybe "scores" depending on your API
            target_lnpdfs,
            jnp.zeros_like(target_lnpdfs),
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