from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from gymnasium import spaces
from ml4ps.h2mg import H2MG, H2MGStructure, H2MGSpace, H2MGNormalizer, h2mg_normal_sample, h2mg_normal_logprob
from ml4ps.reinforcement.policy.base import BasePolicy
from .utils import add_prefix, combine_space

def shallow_repr(h2mg: H2MG) -> Dict[str, Any]:
    results = {}
    for obj_name, hyper_edge in h2mg.hyper_edges.items():
        for feature_name, feature_value in hyper_edge.features.items():
            results[obj_name + "_" + feature_name] = feature_value
    return results

class ContinuousPolicy(BasePolicy):
    """ Continuous policy for power system control.

        Attributes
            observation_space spaces.Space
            action_space spaces.Space
            normalizer ml4ps.Normalizer preprocess input
            postprocessor postprocess output parameters
            nn produce ditribution parameters from observation input.
    """

    def __init__(self, env, normalizer=None, normalizer_args=None, nn_type="h2mgnode", box_to_sigma_ratio=8, **nn_args) -> None:
        # TODO save normalizer, action, obs space, nn args
        self.box_to_sigma_ratio = box_to_sigma_ratio
        self.nn_args = nn_args
        if isinstance(env, gymnasium.vector.VectorEnv):
            self.action_space = env.single_action_space.continuous
        else:
            self.action_space = env.action_space.continuous


        if normalizer is None:
            self.normalizer = self._build_normalizer(env, normalizer_args=normalizer_args)
        else:
            self.normalizer = normalizer

        self.mu_structure = self.action_space.add_suffix("_mu").structure
        self.log_sigma_structure = self.action_space.add_suffix("_log_sigma").structure

        self.nn_output_structure = self.mu_structure.combine(self.log_sigma_structure)
        self.nn = ml4ps.neural_network.get(nn_type, output_structure=self.nn_output_structure)
        self.mu_0, self.log_sigma_0 = self._build_postprocessor(self.action_space)

    def _build_postprocessor(self, h2mg_space: H2MGSpace):
        high = h2mg_space.high
        low = h2mg_space.low
        mu_0 = H2MG.from_structure(self.mu_structure)
        mu_0.flat_array = (high.flat_array + low.flat_array) / 2.
        log_sigma_0 = H2MG.from_structure(self.mu_structure)
        log_sigma_0.flat_array = jnp.log((high.flat_array - low.flat_array) / self.box_to_sigma_ratio)
        return mu_0, log_sigma_0

    def _postprocess_distrib_params(self, distrib_params: H2MG):
        mu = distrib_params.extract_from_structure(self.mu_structure)
        log_sigma = distrib_params.extract_from_structure(self.log_sigma_structure)
        mu_norm = H2MG.from_structure(self.action_space.structure)
        mu_norm.flat_array = jnp.exp(self.log_sigma_0.flat_array) * mu.flat_array + self.mu_0.flat_array
        log_sigma_norm = H2MG.from_structure(self.action_space.structure)
        log_sigma_norm.flat_array = log_sigma.flat_array + self.log_sigma_0.flat_array
        return mu_norm, log_sigma_norm

        # mu, log_sigma = distrib_params.extract_from_structure(self.mu_structure), distrib_params[..., 1]
        # mu_norm = self.log_sigma_0.exp() * mu + self.mu_0
        # log_sigma_norm = log_sigma + self.log_sigma_0
        # return mu_norm, log_sigma_norm

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def _check_valid_action(self, action):
        # TODO
        pass

    def log_prob(self, params, observation, action):
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        mu_norm, log_sigma_norm = self._postprocess_distrib_params(distrib_params)
        return h2mg_normal_logprob(action, mu_norm, log_sigma_norm)

    def sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        mu_norm, log_sigma_norm = self._postprocess_distrib_params(distrib_params)
        if n_action <= 1:
            action = h2mg_normal_sample(rng, mu_norm, log_sigma_norm, deterministic=deterministic)
            log_prob = h2mg_normal_logprob(action, mu_norm, log_sigma_norm)
        else:
            action = [h2mg_normal_sample(rng, mu_norm, log_sigma_norm, deterministic=deterministic) for rng in jax.random.split(rng, n_action)]
            log_prob = [h2mg_normal_logprob(a, mu_norm, log_sigma_norm) for a in action]
        info = self.compute_info(mu_norm, log_sigma_norm)
        return action, log_prob, info

    def compute_info(self, mu_norm: H2MG, log_sigma_norm: H2MG) -> Dict[str, float]:
        mu_norm.add_suffix("_mu")
        log_sigma_norm.add_suffix("_log_sigma")
        info = shallow_repr(mu_norm.apply(lambda x: jnp.asarray(jnp.nanmean(x))))
        info = info | shallow_repr(log_sigma_norm.apply(lambda x: jnp.asarray(jnp.mean(x))))
        return info

    def _build_normalizer(self, env, normalizer_args=None):
        if isinstance(env, gymnasium.vector.VectorEnv):
            backend = env.get_attr("backend")[0]
            data_dir = env.get_attr("data_dir")[0]
            observation_structure = env.get_attr("observation_structure")[0]
        else:
            backend = env.backend
            data_dir = env.data_dir
            observation_structure = env.observation_structure

        if normalizer_args is None:
            return H2MGNormalizer(backend=backend, structure=observation_structure, data_dir=data_dir)
        else:
            return H2MGNormalizer(backend=backend, structure=observation_structure, data_dir=data_dir, **normalizer_args)
