import jax
import jax.numpy as jnp
import numpy as np
from ml4ps.h2mg.core import H2MG, empty_like, map_to_features

def split_rng_like(rng, h2mg: H2MG) -> H2MG:
    """Splits a `jax.random.PRNGKey` according to the structure of the provided H2MG."""
    rng_h2mg = empty_like(h2mg)
    for key, obj_name, feat_name, value in h2mg.local_features_iterator:
        rng, subkey = jax.random.split(rng)
        rng_h2mg[key][obj_name][feat_name] = subkey
    for key, feat_name, value in h2mg.global_features_iterator:
        rng, subkey = jax.random.split(rng)
        rng_h2mg[key][feat_name] = subkey
    return rng_h2mg

def uniform_like(rng, h2mg: H2MG) -> H2MG:
    r"""Draws a H2MG with same structure as provided `h2mg`, with features sampled from \mathcal{U}([0,1])."""
    flat_h2mg = h2mg.flatten()
    x = jax.random.uniform(rng, flat_h2mg.shape)
    return h2mg.unflatten_like(x)

def normal_like(rng, h2mg: H2MG) -> H2MG:
    r"""Draws a H2MG with same structure as provided `h2mg`, with features sampled from \mathcal{N}(0,I)."""
    flat_h2mg = h2mg.flatten()
    x = jax.random.normal(rng, flat_h2mg.shape)
    return h2mg.unflatten_like(x)

def normal_logprob(h2mg: H2MG, mu: H2MG, log_sigma: H2MG) -> float:
    r"""Returns the log-probability of `h2mg`, assuming it is sampled from :math:`\mathcal{N}(mu, diag(\exp(2 log\_sigma)))`.

    .. math::
        \log(P(h2mg | mu, \log(\sigma)) = \sum_{n=1}^N \left( - \frac{\log(2 \pi)}{2} - log\_sigma_n
        - \frac{(h2mg_n - mu_n)^2}{\exp(2 log\_sigma_n)} \right)

    Where $k$ is the amount of features defined in `h2mg`, `mu` and `log_sigma`.
    Fictitious objects (which are represented as NaNs) are not included in the sum.
    Differentiable w.r.t. both `mu` and `sigma`.
    """
    cst = - 0.5 * jnp.log(2 * np.pi)
    return (cst - log_sigma - 0.5 * (-2 * log_sigma).exp() * (jax.lax.stop_gradient(h2mg) - mu)**2).nansum()

def categorical(rng, logits: H2MG, deterministic=False) -> H2MG:
    """Draws a one-hot H2MG with same structure as `logits`; the activated feature is sampled according to `logits`."""
    flat_logits = logits.flatten()
    if deterministic:
        idx = jnp.argmax(jnp.nan_to_num(flat_logits, nan=-jnp.inf))
    else:
        idx = jax.random.categorical(key=rng, logits=jnp.nan_to_num(flat_logits, nan=-jnp.inf))
    flat_res_action = jnp.zeros_like(flat_logits)
    flat_res_action = flat_res_action.at[idx].set(1) + flat_logits * 0.
    res_action = logits.unflatten_like(flat_res_action)
    return res_action

def categorical_logprob(x_onehot: H2MG, logits: H2MG) -> float:
    r"""Returns the log-probability of `h2mg`, assuming it is sampled from a categorical distribution.

    .. math::
        \log(P(h2mg | logits) = \frac{logits^\top . h2mg}{1^\top.h2mg}

    The vector product between two H2MGs with strictly identical structures is performed by multiplying features
    element-wise, and summing over all features.
    Fictitious objects (which are represented as NaNs) are not included in the sum.
    Differentiable w.r.t. `logits`.
    """
    flat_onehot = jax.lax.stop_gradient(x_onehot.flatten())
    flat_logits = jnp.nan_to_num(logits.flatten(), nan=-jnp.inf)
    logits = jnp.nan_to_num(jax.nn.log_softmax(flat_logits), neginf=0.)
    logits_selected = jnp.where(jnp.isnan(flat_onehot), jnp.nan, logits * jnp.nan_to_num(flat_onehot, nan=0.))
    return jnp.nansum(logits_selected)

def categorical_per_feature(rng, logits:H2MG, deterministic=False) -> H2MG:
    """Draws a H2MG where each feature of each hyper-edge follows an independent categorical distribution.

    Notice that while `logits` should have a dimension > 1 for each of its hyper-edge's features, the output
    will be a scalar quantity at each feature of each hyper-edge.

    Examples:
        Let us consider the case of an H2MG with one global variable 'stop', and three shunts with a 'step' feature.

        >>> import jax
        >>> import jax.numpy as jnp
        >>> import ml4ps
        >>> key = jax.random.PRNGKey(0)
        >>> logits = H2MG({'global_features': {'stop': jnp.array([[1., 2.]])},
        >>>                'local_features': {'shunt': {'step': jnp.array([[3., 4., 5.], [6., 7., 8.], [9., 10., 11.]])}}})
        >>> h2mg = ml4ps.h2mg.random.categorical_per_feature(rng, logits)
        >>> h2mg
        {'global_features': {'stop': [1.]}, 'local_features': {'shunt': {'step': [2., 0., 0.]}}}

        The global variable 'stop' is sampled from {0,1}, while the step variable of each shunt is sampled from {0,1,2}.
    """
    rng_h2mg = split_rng_like(rng, logits)
    logits_clean = logits.apply(lambda x: jnp.nan_to_num(x, nan=-jnp.inf))
    if deterministic:
        return map_to_features(jnp.argmax, [logits_clean])
    r = map_to_features(jax.random.categorical, [rng_h2mg, logits_clean])
    return r + logits[:, 0] * 0

def categorical_per_feature_logprob(x_idx: H2MG, logits: H2MG) -> float:
    r"""Returns the log-probability of `h2mg`, assuming it is sampled from a categorical distribution.

    .. math::
        \log(P(h2mg | logits) = \frac{logits^\top . h2mg}{1^\top.h2mg}

    The vector product between two H2MGs with strictly identical structures is performed by multiplying features
    element-wise, and summing over all features.
    Fictitious objects (which are represented as NaNs) are not included in the sum.
    Differentiable w.r.t. `logits`.
    """
    logits= (logits).apply(jax.nn.log_softmax)
    selected_logits = map_to_features(lambda logits, x_i: jnp.take_along_axis(jnp.nan_to_num(logits, nan=0), jnp.expand_dims(x_i.astype(jnp.int32),1), axis=-1), [logits, x_idx])
    return selected_logits.nansum()

