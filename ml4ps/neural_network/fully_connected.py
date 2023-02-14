import jax.numpy as jnp
from flax import linen as nn
from dataclasses import field
from typing import Sequence
from jax.tree_util import tree_flatten


def build_output_dim(x, output_feature_names):
    """Counts the output dimensions by combining the amount of objects in `x`, and the corresponding output features."""
    out_dim = 0
    if "global_features" in output_feature_names:
        for _ in output_feature_names["global_features"]:
            out_dim += 1
    if "local_features" in output_feature_names:
        for k in output_feature_names["local_features"]:
            n_obj = jnp.shape(list(x["local_addresses"][k].values())[0])[0]
            for _ in output_feature_names["local_features"][k]:
                out_dim += n_obj
    return out_dim


def unflatten(h, x, output_feature_names):
    """Transforms h into a h2mg with features defined in `output_feature_names` and amount of objects defined in `x`.

    Note:
        Replaces fictitious objects with nan, while allowing to back-propagate.
    """
    y = {}
    i = 0
    if "global_features" in output_feature_names:
        y["global_features"] = {}
        for k in output_feature_names["global_features"]:
            y["global_features"][k] = h[i:i + 1]
            i += 1
    if "local_features" in output_feature_names:
        y["local_features"] = {}
        for k in output_feature_names["local_features"]:
            y["local_features"][k] = {}
            n_obj = jnp.shape(list(x["local_addresses"][k].values())[0])[0]
            mask = jnp.isnan(list(x["local_addresses"][k].values())[0])
            for f in output_feature_names["local_features"][k]:
                y["local_features"][k][f] = jnp.where(mask, jnp.nan, h[i:i + n_obj])
                i += n_obj
    return y


class FullyConnected(nn.Module):
    """Fully Connected Neural Network.

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of hidden sizes of the MLP.
        output_feature_names (:obj:`dict`): Dictionary of features that the Fully Connected should output.
    """

    hidden_size: Sequence[int] = field(default_factory=lambda: [128])
    output_feature_names: dict = None

    @nn.compact
    def __call__(self, x):
        out_dim = build_output_dim(x, self.output_feature_names)
        h = jnp.concatenate(tree_flatten(x)[0])
        h = jnp.nan_to_num(h, nan=-1)
        for i, d in enumerate(self.hidden_size):
            h = nn.Dense(d)(h)
            h = nn.tanh(h)
        h = nn.Dense(out_dim)(h)
        y = unflatten(h, x, self.output_feature_names)
        return y
