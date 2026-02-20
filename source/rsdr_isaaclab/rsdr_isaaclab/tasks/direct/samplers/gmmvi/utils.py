import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt
import chex
from typing import List
import wandb
import numpy as np
from matplotlib.ticker import MaxNLocator

def reduce_weighted_logsumexp(logx, w=None, axis=None, keep_dims=False, return_sign=False,):
    if w is None:
      lswe = jax.nn.logsumexp(
          logx,
          axis=axis,
          keepdims=keep_dims)

      if return_sign:
        sgn = jnp.ones_like(lswe)
        return lswe, sgn
      return lswe

    log_absw_x = logx + jnp.log(jnp.abs(w))
    max_log_absw_x = jnp.max(
        log_absw_x, axis=axis, keepdims=True,)
    # If the largest element is `-inf` or `inf` then we don't bother subtracting
    # off the max. We do this because otherwise we'd get `inf - inf = NaN`. That
    # this is ok follows from the fact that we're actually free to subtract any
    # value we like, so long as we add it back after taking the `log(sum(...))`.
    max_log_absw_x = jnp.where(
        jnp.isinf(max_log_absw_x),
        jnp.zeros([], max_log_absw_x.dtype),
        max_log_absw_x)
    wx_over_max_absw_x = (jnp.sign(w) * jnp.exp(log_absw_x - max_log_absw_x))
    sum_wx_over_max_absw_x = jnp.sum(
        wx_over_max_absw_x, axis=axis, keepdims=keep_dims)
    if not keep_dims:
      max_log_absw_x = jnp.squeeze(max_log_absw_x, axis)
    sgn = jnp.sign(sum_wx_over_max_absw_x)
    lswe = max_log_absw_x + jnp.log(sgn * sum_wx_over_max_absw_x)
    if return_sign:
      return lswe, sgn
    return lswe

def visualise(log_prob_fn, dr_range_low:chex.Array, dr_range_high : chex.Array, samples: chex.Array = None, eval_samples = None, bijector_log_prob=None, show=False) -> dict:
    plt.close()
    fig = plt.figure()
    ax = fig.add_subplot()
    low, high = dr_range_low, dr_range_high
    x, y = jnp.meshgrid(jnp.linspace(low[0]-0.5, high[0]+0.5, 100), jnp.linspace(low[1]-0.5, high[1]+0.5, 100))
    grid = jnp.c_[x.ravel(), y.ravel()]
    pdf_values = jax.vmap(jnp.exp)(log_prob_fn(sample=grid))
    pdf_values = jnp.reshape(pdf_values, x.shape)
    ctf = plt.contourf(x, y, pdf_values, levels=20, cmap='viridis')
    cbar = fig.colorbar(ctf)
    handles= []
    if samples is not None:
      idx = jax.random.choice(jax.random.PRNGKey(0), samples.shape[0], (300,))
      sample_x = samples[idx,0]
      sample_y = samples[idx,1]
      h1 = ax.scatter(sample_x, sample_y, c='r', alpha=0.5, marker='x')
      handles.append(h1)
    if eval_samples is not None:
      idx = jax.random.choice(jax.random.PRNGKey(0), eval_samples.shape[0], (300,))
      sample_x = eval_samples[idx,0]
      sample_y = eval_samples[idx,1]
      h2 = ax.scatter(sample_x, sample_y, c='b', alpha=0.5, marker='x')
      handles.append(h2)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
    fig.subplots_adjust(right=0.80)
    if handles:
      ax.legend(
          handles=handles,
          loc="center left",
          bbox_to_anchor=(1.02, 0.5),
          frameon=True,
          borderaxespad=0.0,
      )
    fig2 = None
    if bijector_log_prob is not None:
        fig2 = plt.figure()
        ax2 = fig2.add_subplot(111)

        # bijector_log_prob should return log |det J| or something same-shape as logp
        bij_part = bijector_log_prob(grid)              # (10000,)
        bij_part = jnp.reshape(bij_part, x.shape)       # (100,100)

        # model minus bijector term (still jax)
        combined = pdf_values - jax.vmap(jnp.exp)(bij_part)

        combined_np = np.asarray(combined)

        ctf2 = ax2.contourf(x, y, combined_np, levels=20, cmap='viridis')
        cbar2 = fig2.colorbar(ctf2, ax=ax2)
        cbar2.set_label('log p(x) - bijector_term')

        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.set_xlim(float(low[0]), float(high[0]))
        ax2.set_ylim(float(low[1]), float(high[1]))

        ax2.xaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=10, prune=None))
        ax2.minorticks_on()
        ax2.grid(True, which="major", alpha=0.25, linewidth=0.8)
        ax2.grid(True, which="minor", alpha=0.12, linewidth=0.6)
    # plt.xlabel('X')
    # plt.ylabel('Y')
    # plt.xlim(-10, 5)
    # plt.ylim(-5, 5)

    # plt.savefig(os.path.join(project_path('./samples/funnel/'), f"{prefix}funnel.pdf"), bbox_inches='tight', pad_inches=0.1)

    # wb = {"figures/model": [wandb.Image(fig)]}

    if show:
        plt.show()

    return fig, fig2
