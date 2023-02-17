import jax.numpy as jnp
from flax import linen as nn
from dataclasses import field
from typing import Sequence
from jax.tree_util import tree_flatten

from ml4ps.h2mg import H2MG


def build_output_dim(x, feature_dimension):
    """Counts the output dimensions by combining the amount of objects in `x`, and the corresponding output features."""
    out_dim = 0
    if "global_features" in feature_dimension:
        for _, dim in feature_dimension["global_features"].items():
            out_dim += dim
    if "local_features" in feature_dimension:
        for k in feature_dimension["local_features"]:
            n_obj = jnp.shape(list(x["local_addresses"][k].values())[0])[0]
            for _, dim in feature_dimension["local_features"][k].items():
                out_dim += n_obj * dim
    return out_dim


def unflatten(h, x, feature_dimension):
    """Transforms h into a h2mg with features defined in `output_feature_names` and amount of objects defined in `x`.

    Note:
        Replaces fictitious objects with nan, while allowing to back-propagate.
    """
    y = {}
    i = 0
    if "global_features" in feature_dimension:
        y["global_features"] = {}
        for k, dim in feature_dimension["global_features"].items():
            y["global_features"][k] = h[i:i + dim]
            if dim > 1 :
                y["global_features"][k] = jnp.reshape(y["global_features"][k], newshape=[1, dim])
            i += dim
    if "local_features" in feature_dimension:
        y["local_features"] = {}
        for k in feature_dimension["local_features"]:
            y["local_features"][k] = {}
            n_obj = jnp.shape(list(x["local_addresses"][k].values())[0])[0]
            #mask = jnp.isnan(list(x["local_addresses"][k].values())[0])
            mask = list(x["local_addresses"][k].values())[0]
            #values = jnp.expand_dims(values, axis=1)
            for f, dim in feature_dimension["local_features"][k].items():
                y["local_features"][k][f] = h[i:i + n_obj * dim] #jnp.where(mask, jnp.nan, h[i:i + n_obj * dim])
                if dim > 1 :
                    y["local_features"][k][f] = jnp.reshape(y["local_features"][k][f], newshape=[n_obj, dim])
                    y["local_features"][k][f] = jnp.where(jnp.isnan(jnp.expand_dims(mask, axis=1) + 0. * y["local_features"][k][f]), jnp.nan, y["local_features"][k][f])
                else:
                    y["local_features"][k][f] = jnp.where(mask, jnp.nan, h[i:i + n_obj * dim])
                i += n_obj * dim
    return y


class FullyConnected(nn.Module):
    """Fully Connected Neural Network.

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of hidden sizes of the MLP.
        output_feature_names (:obj:`dict`): Dictionary of features that the Fully Connected should output.
    """

    feature_dimension: H2MG
    hidden_size: Sequence[int] = field(default_factory=lambda: [128])
    output_feature_names: dict = None



    @nn.compact
    def __call__(self, x):
        out_dim = build_output_dim(x, self.feature_dimension)
        h = jnp.concatenate(tree_flatten(x)[0])
        h = jnp.nan_to_num(h, nan=-1)
        for i, d in enumerate(self.hidden_size):
            h = nn.Dense(d)(h)
            h = nn.tanh(h)
        h = nn.Dense(out_dim)(h)
        y = unflatten(h, x, self.feature_dimension)
        return H2MG(y)
