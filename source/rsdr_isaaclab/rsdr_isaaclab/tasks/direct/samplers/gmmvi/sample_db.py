from functools import partial
from typing import NamedTuple, Callable
import chex
import jax.numpy as jnp
import jax


class SampleDBState(NamedTuple):
    samples: chex.Array
    means: chex.Array
    chols: chex.Array
    inv_chols: chex.Array
    target_lnpdfs: chex.Array
    target_grads: chex.Array
    mapping: chex.Array
    num_samples_written: chex.Array


class SampleDB(NamedTuple):
    init_sampleDB_state: Callable
    add_samples: Callable
    get_random_sample: Callable
    get_newest_samples: Callable


def setup_sampledb(
    DIM,
    KEEP_SAMPLES,
    MAX_SAMPLES,
    MAX_COMPONENTS,
    DIAGONAL_COVS,
    BATCH_SIZE,
    SAMPLE_SIZE,
    inv_bijector,
    bijector_log_prob,
) -> SampleDB:
    def init_sample_db_state():
        if DIAGONAL_COVS:
            chols = jnp.zeros((MAX_COMPONENTS, DIM))
            inv_chols = jnp.zeros((MAX_COMPONENTS, DIM))
        else:
            chols = jnp.zeros((MAX_COMPONENTS, DIM, DIM))
            inv_chols = jnp.zeros((MAX_COMPONENTS, DIM, DIM))

        return SampleDBState(samples=jnp.zeros((MAX_SAMPLES, DIM)),
                             means=jnp.zeros((MAX_COMPONENTS, DIM)),
                             chols=chols,
                             inv_chols=inv_chols,
                             target_lnpdfs=jnp.zeros(MAX_SAMPLES),
                             target_grads=jnp.zeros((MAX_SAMPLES, DIM)),
                             mapping=jnp.zeros(MAX_SAMPLES, dtype=jnp.int32),
                             num_samples_written=jnp.zeros((1,), dtype=jnp.int32),
                             )

    def add_samples(sampledb_state: SampleDBState, new_samples, new_means, new_chols, new_target_lnpdfs,
                    new_target_grads, new_mapping):
        num_samples_written = sampledb_state.num_samples_written + jnp.shape(new_samples)[0]
        n_new = jnp.shape(new_samples)[0]

        finite_mask = jnp.isfinite(new_target_lnpdfs)  # [n_new]
        finite_mask_exp = finite_mask[:, None]         # [n_new, 1]

        if KEEP_SAMPLES:
            # Roll the buffer and then selectively overwrite only finite rows in the front window.
            front_samples_prev = sampledb_state.samples[:n_new]
            front_lnpdfs_prev = sampledb_state.target_lnpdfs[:n_new]
            front_grads_prev = sampledb_state.target_grads[:n_new]
            front_mapping_prev = sampledb_state.mapping[:n_new]

            samples = jnp.roll(sampledb_state.samples, n_new, axis=0)
            target_lnpdfs = jnp.roll(sampledb_state.target_lnpdfs, n_new, axis=0)
            target_grads = jnp.roll(sampledb_state.target_grads, n_new, axis=0)
            mapping = jnp.roll(sampledb_state.mapping, n_new, axis=0)


            # Keep previous row when not finite.
            front_samples = jnp.where(finite_mask_exp, new_samples, front_samples_prev)
            front_lnpdfs = jnp.where(finite_mask, new_target_lnpdfs, front_lnpdfs_prev)
            front_grads = jnp.where(finite_mask_exp, new_target_grads, front_grads_prev)
            front_mapping = jnp.where(finite_mask, new_mapping, front_mapping_prev)

            samples = samples.at[:n_new].set(front_samples)
            target_lnpdfs = target_lnpdfs.at[:n_new].set(front_lnpdfs)
            target_grads = target_grads.at[:n_new].set(front_grads)
            mapping = mapping.at[:n_new].set(front_mapping)

            means = new_means
            chols = new_chols
            inv_chols = jnp.linalg.inv(new_chols)
        else:
            # Overwrite only the first n_new window; keep old content for invalid rows.
            samples = sampledb_state.samples
            target_lnpdfs = sampledb_state.target_lnpdfs
            target_grads = sampledb_state.target_grads
            mapping = sampledb_state.mapping

            prev_samples = samples[:n_new]
            prev_lnpdfs = target_lnpdfs[:n_new]
            prev_grads = target_grads[:n_new]
            prev_mapping = mapping[:n_new]

            new_front_samples = jnp.where(finite_mask_exp, new_samples, prev_samples)
            new_front_lnpdfs = jnp.where(finite_mask, new_target_lnpdfs, prev_lnpdfs)
            new_front_grads = jnp.where(finite_mask_exp, new_target_grads, prev_grads)
            new_front_mapping = jnp.where(finite_mask, new_mapping, prev_mapping)

            samples = samples.at[:n_new].set(new_front_samples)
            target_lnpdfs = target_lnpdfs.at[:n_new].set(new_front_lnpdfs)
            target_grads = target_grads.at[:n_new].set(new_front_grads)
            mapping = mapping.at[:n_new].set(new_front_mapping)

            means = new_means
            chols = new_chols
            inv_chols = jnp.linalg.inv(new_chols)

        return SampleDBState(
            num_samples_written=num_samples_written,
            samples=samples,
            target_lnpdfs=target_lnpdfs,
            target_grads=target_grads,
            mapping=mapping,
            means=means,
            chols=chols,
            inv_chols=inv_chols,
        )
    def get_random_sample(sample_db_state: SampleDBState, N: int, seed: chex.PRNGKey):
        chosen_indices = jax.random.permutation(seed, jnp.arange(jnp.shape(sample_db_state.samples)[0]),
                                                independent=True)[:N]
        # chosen_indices = Randomness.get_next_random()
        return sample_db_state.samples[chosen_indices], sample_db_state.target_lnpdfs[chosen_indices]

    def _gaussian_log_pdf(mean, chol, inv_chol, x):
        jac_log_prob = bijector_log_prob(x)
        x = inv_bijector(x)
        if DIAGONAL_COVS:
            constant_part = - 0.5 * DIM * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(chol))
            base = constant_part - 0.5 * jnp.sum(
                jnp.square(jnp.expand_dims(inv_chol, 1) * jnp.transpose(jnp.expand_dims(mean, 0) - x)),
                axis=0,
            )
            return base + jac_log_prob
        else:
            constant_part = - 0.5 * DIM * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(jnp.diag(chol)))
            base = constant_part - 0.5 * jnp.sum(jnp.square(inv_chol @ jnp.transpose(mean - x)), axis=0)
            return base + jac_log_prob
    @partial(jax.jit, static_argnames=('N',))
    def get_newest_samples_deprecated(sampledb_state: SampleDBState, N):
        def _compute_log_pdfs(sampledb_state, component_id, sample):
            return jax.lax.cond(component_id == -1,
                                lambda: jnp.full(sample.shape[0], -jnp.inf),
                                lambda: _gaussian_log_pdf(sampledb_state.means[component_id],
                                                          sampledb_state.chols[component_id],
                                                          sampledb_state.inv_chols[component_id], sample))

        active_sample = sampledb_state.samples[:N]
        active_target_lnpdfs = sampledb_state.target_lnpdfs[:N]
        active_target_grads = sampledb_state.target_grads[:N]
        active_mapping = sampledb_state.mapping[:N]
        @jax.jit
        def compute_background_pdf():
            active_components, count = jnp.unique(active_mapping, return_counts=True, size=sampledb_state.means.shape[0], fill_value=-1)
            weights = count / jnp.sum(count)
            return jax.nn.logsumexp(jax.vmap(_compute_log_pdfs, in_axes=(None, 0, None))(sampledb_state, active_components, active_sample) + jnp.expand_dims(jnp.log(weights), 1), axis=0)
        log_pdfs = compute_background_pdf()

        return log_pdfs, active_sample, active_mapping, active_target_lnpdfs, active_target_grads

    return SampleDB(init_sampleDB_state=init_sample_db_state,
                    add_samples=add_samples,
                    get_random_sample=get_random_sample,
                    get_newest_samples=get_newest_samples_deprecated,
                    )
