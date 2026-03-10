import functools
from learning.module.gmmvi.configs import get_default_algorithm_config, update_config
from learning.module.gmmvi.gmm_setup import setup_diagonal_gmm, setup_full_cov_gmm, GMMWrapperState, setup_gmm_wrapper
from algorithms.gmmvi.optimization.gmmvi_modules.component_stepsize_adaptation import (
    setup_improvement_based_stepsize_adaptation,
    setup_decaying_component_stepsize_adaptation, setup_fixed_component_stepsize_adaptation)
from algorithms.gmmvi.optimization.gmmvi_modules.weight_stepsize_adaptation import (
    WeightStepsizeAdaptationState, setup_fixed_weight_stepsize_adaptation, setup_decaying_weight_stepsize_adaptation,
    setup_improvement_based_weight_stepsize_adaptation)
from learning.module.gmmvi.sample_db import SampleDBState, setup_sampledb
import jax
import jax.numpy as jnp
from typing import NamedTuple, Callable
import chex
from learning.module.gmmvi.ng_update import get_ng_update_fns
from learning.module.gmmvi.stepsize_update import setup_weight_stepsize_fn, update_component_stepsize_adaptive, update_component_stepsize_decaying
from learning.module.gmmvi.weight_update import setup_weight_update_fn
from learning.module.gmmvi.sample_selector import setup_fixed_sample_selector
from learning.module.gmmvi.component_adaptation import ComponentAdaptationState, setup_vips_component_adaptation


class TrainState(NamedTuple):
    temperature: float
    num_updates: chex.Array
    model_state: GMMWrapperState
    sample_db_state: SampleDBState
    component_adaptation_state: ComponentAdaptationState


class GMMVI(NamedTuple):
    initial_train_state: TrainState
    train_iter: Callable
    eval: Callable


def setup_gmmvi(config, seed):
    dim= config.dim
    # dim = target.dim
    # target_log_prob = target.log_prob

    # get necessary config entries
    config = update_config(update_config(get_default_algorithm_config(config.algorithm.algorithm), config), config.algorithm)
    # setup GMM
    if config["model_initialization"]["use_diagonal_covs"]:
        gmm = setup_diagonal_gmm(dim)
    else:
        gmm = setup_full_cov_gmm(dim)
    gmm_state = gmm.init_gmm_state(seed,
                                   config["model_initialization"]["num_initial_components"],
                                   config["model_initialization"]["prior_mean"],
                                   config["model_initialization"]["prior_scale"],
                                   config["model_initialization"]["use_diagonal_covs"],
                                   config["model_initialization"]["prior_scale"] ** 2)

    if "initial_l2_regularizer" in config["ng_estimator_config"]:
        initial_l2_regularizer = config["ng_estimator_config"]['initial_l2_regularizer']
    else:
        initial_l2_regularizer = 1e-12

    # setup GMMWrapper
    model = setup_gmm_wrapper(gmm,
                              config["component_stepsize_adapter_config"]["initial_stepsize"],
                              initial_l2_regularizer,
                              10000)
    model_state = model.init_gmm_wrapper_state(gmm_state)

    # setup SampleDB
    sample_db = setup_sampledb(dim,
                               config["use_sample_database"],
                               config["max_database_size"],
                               config["model_initialization"]["use_diagonal_covs"],
                               config['sample_selector_config']["desired_samples_per_component"])
    sample_db_state = sample_db.init_sampleDB_state()
    if config["num_component_adapter_type"] == "adaptive":
        component_adapter = setup_vips_component_adaptation(sample_db,
                    model,
                    dim,
                    config["model_initialization"]["prior_mean"],
                    config["model_initialization"]["prior_scale"] ** 2,
                    config["model_initialization"]["use_diagonal_covs"],
                    config["num_component_adapter_config"]["del_iters"],
                    config["num_component_adapter_config"]["add_iters"],
                    config["num_component_adapter_config"]["max_components"],
                    config["num_component_adapter_config"]["thresholds_for_add_heuristic"],
                    config["num_component_adapter_config"]["min_weight_for_del_heuristic"],
                    config["num_component_adapter_config"]["num_database_samples"],
                    config["num_component_adapter_config"]["num_prior_samples"],
                    )
        component_adapter_state = component_adapter.init_component_adaptation()

    # setup Component StepsizeAdaptation
    if config["component_stepsize_adapter_type"] == "fixed":
        component_stepsize_fn = lambda gmm_wrapper_state: gmm_wrapper_state.stepsizes
    elif config["component_stepsize_adapter_type"] == "decaying":
        component_stepsize_fn = functools.partial(update_component_stepsize_decaying, 
                                                                INITIAL_STEPSIZE=config["component_stepsize_adapter_config"]["initial_stepsize"],
                                                                ANNEALING_EXPONENT=config["component_stepsize_adapter_config"]["annealing_exponent"])
    elif config["component_stepsize_adapter_type"] == "adaptive":
        component_stepsize_fn = functools.partial(update_component_stepsize_adaptive, 
                                                                MIN_STEPSIZE=config["component_stepsize_adapter_config"]["min_stepsize"],               
                                                                MAX_STEPSIZE=config["component_stepsize_adapter_config"]["max_stepsize"],
                                                                STEPSIZE_INC_FACTOR=config["component_stepsize_adapter_config"]["stepsize_inc_factor"],
                                                                STEPSIZE_DEC_FACTOR=config["component_stepsize_adapter_config"]["stepsize_dec_factor"])
    # Choose just fixed sample selector for now
    sample_selector = setup_fixed_sample_selector(sample_db,
                                                    model,
                                                    config['sample_selector_config']["desired_samples_per_component"],
                                                    config['sample_selector_config']["ratio_reused_samples_to_desired"])
    # Choose jsut direct weight updater for now
    weight_update_fn = setup_weight_update_fn(model,
                                                config['temperature'],
                                                config["weight_updater_config"]["use_self_normalized_importance_weights"])
    
    # Choose just fixed weight stepsize adaptation

    ng_update_fn, hass_grad_fn = get_ng_update_fns(model,
                                                dim,
                                                config["model_initialization"]["use_diagonal_covs"],
                                                config['ng_estimator_config']["use_self_normalized_importance_weights"],
                                                config['temperature'],
                                                initial_l2_regularizer,
                                                )
    def train_iter(train_state: TrainState, key: chex.Array):

        key, subkey = jax.random.split(key)
        new_sample_db_state, samples, mapping, sample_dist_densities, target_lnpdfs, target_lnpdf_grads =\
              sample_selector.select_samples(train_state.model_state,
                                            train_state.sample_db_state,
                                            subkey)
        new_component_stepsizes = component_stepsize_fn(train_state.model_state)
        new_model_state = model.update_stepsizes(train_state.model_state, new_component_stepsizes)
        expected_hessian_neg, expected_grad_neg = hass_grad_fn(new_model_state,
                                                                samples,
                                                                mapping,
                                                                sample_dist_densities,
                                                                target_lnpdfs,
                                                                target_lnpdf_grads,
                                                                int(train_state.model_state.gmm_state.num_components))
        new_model_state = ng_update_fn(new_model_state,
                                        expected_hessian_neg,
                                        expected_grad_neg,
                                        new_model_state.stepsizes)

        # new_weight_stepsize_adapter_state = weight_stepsize_adapter.update_stepsize(train_state.weight_stepsize_adapter_state, new_model_state)
        new_model_state = weight_update_fn(new_model_state, samples, sample_dist_densities, target_lnpdfs,
                                                        config['weight_stepsize_adapter_config']["initial_stepsize"])
        new_num_updates = train_state.num_updates + 1
        key, subkey = jax.random.split(key)
        if config["num_component_adapter_type"] == "adaptive":
            new_model_state, new_component_adapter_state, new_sample_db_state = \
            component_adapter.adapt_number_of_components(train_state.component_adaptation_state,
                                                        new_sample_db_state,
                                                        new_model_state,
                                                        new_num_updates,
                                                        subkey)
        else:
            new_component_adapter_state = train_state.component_adaptation_state
        return TrainState(temperature=train_state.temperature,
                          model_state=new_model_state,
                          component_adaptation_state=new_component_adapter_state,
                          num_updates=new_num_updates,
                          sample_db_state=new_sample_db_state)

    def eval(seed: chex.Array, train_state: TrainState, target_samples=None):
        samples = model.sample(train_state.model_state.gmm_state, seed, config["eval_samples"])[0]
        log_prob_model = jax.vmap(model.log_density, in_axes=(None, 0))(train_state.model_state.gmm_state, samples)
        log_prob_target = jax.vmap(target.log_prob)(samples)
        log_ratio = log_prob_target - log_prob_model

        if target_samples is not None:
            fwd_log_prob_model = jax.vmap(model.log_density, in_axes=(None, 0))(train_state.model_state.gmm_state, target_samples)
            fwd_log_prob_target = jax.vmap(target.log_prob)(target_samples)
            fwd_log_ratio = fwd_log_prob_target - fwd_log_prob_model
        else:
            fwd_log_ratio = None

        return samples, log_ratio, log_prob_target, fwd_log_ratio

    initial_train_state = TrainState(temperature=config['temperature'],
                                     num_updates=jnp.array([0]),
                                     model_state=model_state,
                                     sample_db_state=sample_db_state,
                                     component_adaptation_state=component_adapter_state)
    return initial_train_state, train_iter, eval
