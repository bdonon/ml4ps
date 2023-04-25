from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, List
import json
import pickle
from numbers import Number


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

    def __init__(self, env, normalizer=None, normalizer_args=None, nn_type="h2mgnode", box_to_sigma_ratio=8, file=None, cst_sigma=None, clip_sigma=None, nn_args={}) -> None:
        if file is not None:
            self.load(file)
        # TODO save normalizer, action, obs space, nn args
        self.box_to_sigma_ratio = box_to_sigma_ratio
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

        nn_output_structure = self.mu_structure.combine(self.log_sigma_structure)
        self.nn = ml4ps.neural_network.get(nn_type, output_structure=nn_output_structure, **nn_args)
        self.mu_0, self.log_sigma_0 = self._build_postprocessor(self.action_space)

        self.cst_sigma = cst_sigma
        self.clip_sigma = clip_sigma

    def _build_postprocessor(self, h2mg_space: H2MGSpace):
        high = h2mg_space.high
        low = h2mg_space.low
        mu_0 = H2MG.from_structure(self.mu_structure)
        mu_0.flat_array = (high.flat_array + low.flat_array) / 2.
        log_sigma_0 = H2MG.from_structure(self.mu_structure)
        log_sigma_0.flat_array = jnp.log((high.flat_array - low.flat_array) / self.box_to_sigma_ratio)
        return mu_0, log_sigma_0
    
    def _clip_log_sigma(self, log_sigma, eps=0.01):
        # return jax.numpy.clip(log_sigma, a_min=jnp.log(self.clip_sigma))
        return jax.numpy.clip((1-eps)*log_sigma, a_min=jnp.log(self.clip_sigma)) + eps*log_sigma # soft clipping

    def _postprocess_distrib_params(self, distrib_params: H2MG):
        mu = distrib_params.extract_from_structure(self.mu_structure)
        log_sigma = distrib_params.extract_from_structure(self.log_sigma_structure)
        mu_norm = H2MG.from_structure(self.action_space.structure)
        mu_norm.flat_array = jnp.exp(self.log_sigma_0.flat_array) * mu.flat_array + self.mu_0.flat_array
        log_sigma_norm = H2MG.from_structure(self.action_space.structure)
        log_sigma_norm.flat_array = log_sigma.flat_array + self.log_sigma_0.flat_array
        if self.clip_sigma is not None and isinstance(self.clip_sigma, Number):
            log_sigma_norm.flat_array = self._clip_log_sigma(log_sigma_norm.flat_array)
        if self.cst_sigma is not None and isinstance(self.cst_sigma, Number):
            log_sigma_norm.flat_array = jnp.full_like(log_sigma.flat_array, jnp.log(self.cst_sigma)) # constant sigma
        # print(log_sigma_norm.flat_array))
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

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.box_to_sigma_ratio, f)
            pickle.dump(self.action_space, f)
            pickle.dump(self.normalizer, f)
            pickle.dump(self.mu_structure, f)
            pickle.dump(self.log_sigma_structure, f)
            pickle.dump(self.nn, f)
            pickle.dump(self.mu_0, f)
            pickle.dump(self.log_sigma_0, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.box_to_sigma_ratio = pickle.load(f)
            self.action_space = pickle.load(f)
            self.normalizer = pickle.load(f)
            self.mu_structure = pickle.load(f)
            self.log_sigma_structure = pickle.load(f)
            self.nn = pickle.load(f)
            self.mu_0 = pickle.load(f)
            self.log_sigma_0 = pickle.load(f)
