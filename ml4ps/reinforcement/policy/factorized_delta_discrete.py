from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from gymnasium import spaces
from ml4ps import Normalizer, h2mg
from ml4ps.reinforcement.policy.base import BasePolicy

from .utils import (add_prefix, combine_feature_names, flatten_dict,
                    slice_with_prefix, space_to_feature_names, unflatten_like, get_single_action_space)


class FactorizedDeltaDiscrete(BasePolicy):

    def __init__(self, env, normalizer=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, **nn_args):
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.normalizer = normalizer or self.build_normalizer(env, normalizer_args)
        action_space = get_single_action_space(env)
        self.multi_discrete = action_space.multi_discrete.feature_dimension * 3
        self.multi_binary = action_space.multi_binary.feature_dimension * 2
        self.nn = ml4ps.neural_network.get(nn_type, {
            "feature_dimension": self.multi_binary.combine(self.multi_discrete), **nn_args})

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def _check_valid_action(self, action):
        pass

    def log_prob(self, params, observation, action):
        norm_observation = self.normalizer(observation)
        logits = self.nn.apply(params, norm_observation)
        return h2mg.categorical_per_feature_logprob(action, logits)

    def sample(self, params, observation, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        norm_observation = self.normalizer(observation)
        logits = self.nn.apply(params, norm_observation)
        if n_action <= 1:
            action = h2mg.categorical_per_feature(rng, logits, deterministic=deterministic)
            log_prob = h2mg.categorical_per_feature_logprob(action, logits)
        else:
            action = [h2mg.categorical_per_feature(rng, logits, deterministic=deterministic) for rng in
                jax.random.split(rng, n_action)]
            log_prob = [h2mg.categorical_per_feature_logprob(a, logits) for a in action]
        info = h2mg.shallow_repr(h2mg.map_to_features(lambda x: jnp.asarray(jnp.mean(x)), [logits]))
        return action, log_prob, info

    def build_normalizer(self, env, normalizer_args=None, data_dir=None):
        if isinstance(env, gymnasium.vector.VectorEnv):
            backend = env.get_attr("backend")[0]
            data_dir = env.get_attr("data_dir")[0]
        else:
            backend = env.backend
            data_dir = env.data_dir

        if normalizer_args is None:
            return Normalizer(backend=backend, data_dir=data_dir)
        else:
            return Normalizer(backend=backend, data_dir=data_dir,
                              **normalizer_args)  # TODO kwargs.get("normalizer_args", {})
