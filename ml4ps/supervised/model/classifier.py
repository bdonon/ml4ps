from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from gymnasium import spaces
from ml4ps.h2mg import H2MG, H2MGNormalizer
from ml4ps.supervised.model.base import BaseModel



class Classifier(BaseModel):
    """
    """

    def __init__(self, problem=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, **nn_args) -> None:
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.input_normalizer = H2MGNormalizer(backend=problem.backend, data_dir=problem.data_dir,
                                               structure=problem.input_structure, **normalizer_args)
        self.output_normalizer = H2MGNormalizer(backend=problem.backend, data_dir=problem.data_dir,
                                                structure=problem.output_structure, **normalizer_args)
        self.nn = ml4ps.neural_network.get(nn_type, structure=problem.output_space.discrete.structure, **nn_args)

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def loss(self, params, x, y):
        x_norm = self.input_normalizer(x)
        y_hat_norm = self.nn.apply(params, x_norm)
        y_norm = self.output_normalizer(y)
        return jnp.nanmean((y_norm.flat_array - y_hat_norm.flat_array)**2)

    def predict(self, params, x):
        x_norm = self.input_normalizer(x)
        y_hat_norm = self.nn.apply(params, x_norm)
        return self.output_normalizer.inverse(y_hat_norm)
