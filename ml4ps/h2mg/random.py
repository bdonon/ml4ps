import jax
import jax.numpy as jnp
from ml4ps.h2mg.core import H2MG, H2MGStructure


def h2mg_uniform_sample(rng, min: float | H2MG = 0, max: float | H2MG = 1., structure: H2MGStructure = None) -> H2MG:
    r"""Draws a H2MG with features sampled from :math:`\mathcal{U}([min, max])`.

    The `min` and `max` arguments can be scalar or H2MG.
    The `structure` argument is optional if either `min` or `max` is an H2MG.

    Note that the log-prob of a uniform distribution is non-differentiable.
    As a consequence, we do not provide any log-prob function implementation for the normal distribution
    """
    if (not isinstance(min, H2MG)) and (not isinstance(max, H2MG)):
        r = H2MG.from_structure(structure)
    elif isinstance(min, H2MG):
        r = H2MG.from_structure(min.structure)
    else:
        r = H2MG.from_structure(max.structure)

    min_array = min.flat_array if isinstance(min, H2MG) else min
    max_array = max.flat_array if isinstance(max, H2MG) else max

    r.flat_array = min_array + jax.random.uniform(rng, r.flat_array.shape) * (max_array - min_array)
    return r


def h2mg_normal_sample(rng, mu: float | H2MG = 0., log_sigma: float | H2MG = 0.,
                       structure: H2MGStructure = None, deterministic: bool = False) -> H2MG:
    r"""Draws a H2MG with features sampled from :math:`\mathcal{N}(mu,exp(log_sigma))`.

    The `mu` and `log_sigma` arguments can be scalar or H2MG.
    The `structure` argument is optional if either `min` or `max` is an H2MG.
    """
    if (not isinstance(mu, H2MG)) and (not isinstance(log_sigma, H2MG)):
        r = H2MG.from_structure(structure)
    elif isinstance(mu, H2MG):
        r = H2MG.from_structure(mu.structure)
    else:
        r = H2MG.from_structure(log_sigma.structure)

    mu_array = mu.flat_array if isinstance(mu, H2MG) else mu
    log_sigma_array = log_sigma.flat_array if isinstance(log_sigma, H2MG) else log_sigma

    if deterministic:
        r.flat_array = mu_array
    else:
        r.flat_array = mu_array + jax.random.normal(rng, r.flat_array.shape) * jnp.exp(log_sigma_array)
    return r


def h2mg_normal_logprob(x: H2MG, mu: float | H2MG = 0., log_sigma: float | H2MG = 0.) -> float:
    r"""Returns the log-probability of `x`, assuming it is sampled from :math:`\mathcal{N}(mu, diag(\exp(2 log\_sigma)))`.

    .. math::
        \log(P(h2mg | mu, \log(\sigma)) = \sum_{n=1}^N \left( - \frac{\log(2 \pi)}{2} - log\_sigma_n
        - \frac{(h2mg_n - mu_n)^2}{\exp(2 log\_sigma_n)} \right)

    Where $N$ is the amount of features defined in `h2mg`, `mu` and `log_sigma`.
    The `mu` and `log_sigma` arguments can be scalar or H2MG.
    Fictitious objects (which are represented as NaNs) are not included in the sum.
    Differentiable w.r.t. both `mu` and `sigma`.
    """
    cst = - 0.5 * jnp.log(2 * jnp.pi)
    x_array = jax.lax.stop_gradient(x.flat_array) if isinstance(x, H2MG) else jax.lax.stop_gradient(x)
    mu_array = mu.flat_array if isinstance(mu, H2MG) else mu
    log_sigma_array = log_sigma.flat_array if isinstance(log_sigma, H2MG) else log_sigma
    return jnp.nansum(cst - log_sigma_array - 0.5 * jnp.exp(-2*log_sigma_array)*(x_array - mu_array)**2)


def h2mg_categorical_sample(rng: jax.random.PRNGKey, logits: H2MG, deterministic: bool = False) -> H2MG:
    """Draws a one-hot H2MG, only one feature for the whole H2MG is set to 1 according to `logits`."""
    sample = H2MG.from_structure(logits.structure, value=0.)
    flat_logits = logits.flat_array
    if deterministic:
        idx = jnp.argmax(jnp.nan_to_num(flat_logits, nan=-jnp.inf))
    else:
        idx = jax.random.categorical(key=rng, logits=jnp.nan_to_num(flat_logits, nan=-jnp.inf))
    sample.flat_array = sample.flat_array.at[idx].set(1)
    return sample


def h2mg_categorical_logprob(x: H2MG, logits: H2MG) -> float:
    r"""Returns the log-probability of `x`, assuming it is sampled from a categorical distribution.

    .. math::
        \log(P(h2mg | logits) = \frac{logits^\top . h2mg}{1^\top.h2mg}

    The vector product between two H2MGs with strictly identical structures is performed by multiplying features
    element-wise, and summing over all features.
    Fictitious objects (which are represented as NaNs) are not included in the sum.
    Differentiable w.r.t. `logits`.
    """
    flat_onehot = jax.lax.stop_gradient(x.flat_array)
    flat_logprobs = jnp.nan_to_num(jax.nn.log_softmax(jnp.nan_to_num(logits.flat_array, nan=-jnp.inf)), neginf=0.)
    selected_logprobs = jnp.where(jnp.isnan(flat_onehot), jnp.nan, flat_logprobs * jnp.nan_to_num(flat_onehot, nan=0.))
    return jnp.nansum(selected_logprobs)


def h2mg_factorized_categorical_sample(rng: jax.random.PRNGKey, logits: H2MG, deterministic: bool = False) -> H2MG:
    """Draws a H2MG where each hyper-edge object has strictly one feature set to 1, following `logits`."""
    sample = H2MG.from_structure(logits.structure)
    for k, hyper_edges in logits.items():
        rng, subkey = jax.random.split(rng)
        if hyper_edges.array is not None:
            logits_array = jnp.nan_to_num(hyper_edges.array, nan=-jnp.inf)
            if deterministic:
                idx = jnp.argmax(logits_array, axis=-1)
            else:
                idx = jax.random.categorical(key=subkey, logits=jnp.nan_to_num(logits_array, nan=-jnp.inf), axis=-1)
            def set_to_one(array, idx):
                return array.at[idx].set(1)
            vmap_set_to_one = jax.jit(jax.vmap(set_to_one, in_axes=(0, 0), out_axes=0))
            sample[k].array = vmap_set_to_one(sample[k].array, idx)
    return sample


def h2mg_factorized_categorical_logprob(x_idx: H2MG, logits: H2MG) -> float:
    """Returns the log-probability of `h2mg`, assuming it is sampled from a factorized categorical distribution."""
    logprob = 0.
    for k, hyper_edges in x_idx.items():
        if hyper_edges.array is not None:
            onehot = jax.lax.stop_gradient(hyper_edges.array)
            logprobs = jnp.nan_to_num(jax.nn.log_softmax(jnp.nan_to_num(logits[k].array, nan=-jnp.inf)), neginf=0.)
            selected_logprobs = jnp.where(jnp.isnan(onehot), jnp.nan, logprobs * jnp.nan_to_num(onehot, nan=0.))
            logprob += jnp.nansum(selected_logprobs)
    return logprob
