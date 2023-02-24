from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from gymnasium import spaces
from ml4ps import Normalizer, h2mg
from ml4ps.supervised.model.base import BaseModel



class Regressor(BaseModel):
    """
    """

    def __init__(self, problem=None, normalizer=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, **nn_args) -> None:
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.normalizer = normalizer or Normalizer(backend=problem.backend, data_dir=problem.data_dir)# TODO, **normalizer_args)
        self.nn = ml4ps.neural_network.get(nn_type, {"feature_dimension":problem.output_space.continuous.feature_dimension, **nn_args})


    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def loss(self, params, x, y):
        x_norm = self.normalizer(x)
        y_hat_norm = self.nn.apply(params, x_norm) / 8.
        y_norm = self.normalizer(y)
        return ((y_hat_norm-y_norm)**2).nansum()

    def predict(self, params, x):
        x_norm = self.normalizer(x)
        y_hat_norm = self.nn.apply(params, x_norm) / 8.
        return self.normalizer.inverse(y_hat_norm)
