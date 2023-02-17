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
from ml4ps.reinforcement.policy.base import BasePolicy



class ContinuousPolicy(BasePolicy):
    """ Continuous policy for power system control.

        Attributes
            observation_space spaces.Space
            action_space spaces.Space
            normalizer ml4ps.Normalizer preprocess input
            postprocessor postprocess output parameters
            nn produce ditribution parameters from observation input.
    """

    def __init__(self, env=None, normalizer=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, box_to_sigma_ratio=8, **nn_args) -> None:
        # TODO save normalizer, action, obs space, nn args
        self.box_to_sigma_ratio = box_to_sigma_ratio
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.normalizer = normalizer or self._build_normalizer(env, normalizer_args)
        self.nn = ml4ps.neural_network.get(nn_type, {"feature_dimension":env.action_space.continuous.feature_dimension * 2, **nn_args})
        self.mu_0, self.log_sigma_0 = self._build_postprocessor(env.action_space.continuous)


    def _build_postprocessor(self, space):
        return (space.high + space.low) / 2., ((space.high - space.low) / self.box_to_sigma_ratio).log()


    def _postprocess_distrib_params(self, distrib_params):
        mu, log_sigma = distrib_params[:, 0], distrib_params[:, 1]
        mu_norm = self.log_sigma_0.exp() * mu + self.mu_0
        log_sigma_norm = log_sigma + self.log_sigma_0
        return mu_norm, log_sigma_norm


    def init(self, rng, obs):
        return self.nn.init(rng, obs)


    def _check_valid_action(self, action):
        # TODO
        pass

    def log_prob(self, params, observation, action):
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        mu_norm, log_sigma_norm = self._postprocess_distrib_params(distrib_params)
        return h2mg.normal_logprob(action, mu_norm, log_sigma_norm)


    def sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        mu_norm, log_sigma_norm = self._postprocess_distrib_params(distrib_params)
        if n_action <= 1:
            action = self._sample_from_distrib_params(rng, mu_norm, log_sigma_norm, deterministic=deterministic)
            log_prob = h2mg.normal_logprob(action, mu_norm, log_sigma_norm)
        else:
            action = [self._sample_from_distrib_params(rng, mu_norm, log_sigma_norm, deterministic=deterministic) for rng in jax.random.split(rng, n_action)]
            log_prob = [h2mg.normal_logprob(a, mu_norm, log_sigma_norm) for a in action]
        info = h2mg.shallow_repr(h2mg.map_to_features(lambda x: jnp.asarray(jnp.mean(x)), [distrib_params]))
        return action, log_prob, info


    def _sample_from_distrib_params(self, rng, mu, log_sigma, deterministic=False):
        if deterministic:
            return mu
        else:
            return mu + h2mg.normal_like(rng, mu) * log_sigma.exp()




    def _build_normalizer(self, env, normalizer_args=None, data_dir=None):
        if isinstance(env, gymnasium.vector.VectorEnv):
            backend = env.get_attr("backend")[0]
            data_dir = env.get_attr("data_dir")[0]
        else:
            backend = env.backend
            data_dir = env.data_dir

        if normalizer_args is None:
            return Normalizer(backend=backend, data_dir=data_dir)
        else:
            return Normalizer(backend=backend, data_dir=data_dir, **normalizer_args) # TODO kwargs.get("normalizer_args", {})
