import pickle
from copy import deepcopy
from functools import partial
from numbers import Number
from time import time
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Sequence, Tuple, Union)

import gymnasium as gym
import jax
import jax.numpy as jnp
import ml4ps
from jax import jit
from jax import numpy as jnp
from jax import vmap
from jax.random import KeyArray, PRNGKey, PRNGKeyArray, split
from jax.tree_util import tree_leaves
from ml4ps import h2mg
from ml4ps.h2mg import (H2MG, H2MGSpace, H2MGStructure, HyperEdgesStructure,
                        h2mg_categorical_logprob, h2mg_categorical_sample,
                        h2mg_normal_logprob, h2mg_normal_sample, shallow_repr,
                        spaces, vmap_h2mg_categorical_entropy)
from ml4ps.neural_network import get as get_neural_network
from ml4ps.reinforcement.policy.base import BasePolicy
from ml4ps.reinforcement.policy.continous_policy import (ContinuousPolicy,
                                                         manual_normalization)
from tqdm import tqdm


class ContinuousAndDiscrete(BasePolicy):
    def __init__(self, env: gym.Env | gym.vector.VectorEnv, normalizer=None, normalizer_args=None, nn_type="h2mgnode",
                 box_to_sigma_ratio=8, cst_sigma=None, clip_sigma=None, file=None, nn_args={}):
        self.nn_args = nn_args
        self.normalizer = normalizer or self._build_normalizer(
            env, normalizer_args)
        if isinstance(env, gym.vector.VectorEnv):
            multi_discrete: h2mg.H2MGSpace = env.single_action_space.multi_discrete
            multi_binary: h2mg.H2MGSpace = env.single_action_space.multi_binary
            self.multi_binary_struct: h2mg.H2MGStructure = env.single_action_space.multi_binary.structure
            self.action_space: h2mg.H2MGSpace = env.single_action_space
        else:
            multi_discrete: h2mg.H2MGSpace = env.action_space.multi_discrete
            multi_binary: h2mg.H2MGStructure = env.action_space.multi_binary
            self.multi_binary_struct: h2mg.H2MGStructure = env.action_space.multi_binary.structure
            self.action_space: h2mg.H2MGSpace = env.action_space
        self.continuous_action_space: h2mg.H2MGSpace = self.action_space.continuous
        self.discrete_action_space: h2mg.H2MGSpace = multi_discrete.combine(
            multi_binary)
        # Discrete init
        self.multi_discrete_struct = multi_discrete.structure
        self.multi_discrete_up_struct = multi_discrete.add_suffix(
            "_up").structure
        self.multi_discrete_down_struct = multi_discrete.add_suffix(
            "_down").structure
        discrete_output_structure = self.multi_binary_struct.combine(
            self.multi_discrete_up_struct).combine(self.multi_discrete_down_struct)
        self.discrete_nn = ml4ps.neural_network.get(nn_type,
                                                    output_structure=discrete_output_structure,
                                                    **nn_args)

        # Continous init
        self.cst_sigma = cst_sigma
        self.clip_sigma = clip_sigma
        self.box_to_sigma_ratio = box_to_sigma_ratio

        self.mu_structure = self.continuous_action_space.add_suffix(
            "_mu").structure
        self.log_sigma_structure = self.continuous_action_space.add_suffix(
            "_log_sigma").structure

        continous_output_structure = self.mu_structure.combine(
            self.log_sigma_structure)
        self.continuous_nn = ml4ps.neural_network.get(
            nn_type, output_structure=continous_output_structure, **nn_args)
        self.mu_0, self.log_sigma_0 = self._build_postprocessor(
            self.continuous_action_space)

    def combine_params(self, discrete_params: Dict, continuous_params: Dict) -> Dict:
        return {"discrete": discrete_params, "continuous": continuous_params}

    def split_params(self, params: Dict) -> Tuple[Dict, Dict]:
        return params["discrete"], params["continuous"]

    def init(self, rng, obs) -> Dict:
        discrete_params = self.discrete_nn.init(rng, obs)
        discrete_action, _, _ = self.discrete_sample(
            discrete_params, obs, rng=rng, deterministic=True, n_action=1)
        continuous_obs = self.combine_obs_and_discrete_action(
            obs, discrete_action)
        continuous_params = self.continuous_nn.init(rng, continuous_obs)
        return self.combine_params(discrete_params, continuous_params)

    def combine_obs_and_discrete_action(self, obs:  H2MG, discrete_action: H2MG) -> H2MG:
        assert(isinstance(obs, H2MG))
        assert(isinstance(discrete_action, H2MG))
        return obs.combine(discrete_action)

    def _check_valid_action(self, action):
        pass

    def log_prob(self, params: Dict, observation: H2MG, action: H2MG) -> Tuple[float, Any]:
        discrete_params, continuous_params = self.split_params(params)
        discrete_log_prob, logits = self.discrete_log_prob(
            discrete_params, observation, action)
        discrete_action = action.extract_from_structure(
            self.discrete_action_space.structure)  # TODO
        continuous_observation = self.combine_obs_and_discrete_action(
            observation, discrete_action)
        continuous_log_prob, continuous_info = self.continuous_log_prob(
            continuous_params, continuous_observation, action)

        return discrete_log_prob + continuous_log_prob, \
            self.combine_params(logits, continuous_info) | {"continuous_log_prob": continuous_log_prob,
                                                            "discrete_log_prob": discrete_log_prob}

    def sample(self, params: dict, observation: dict, rng, deterministic: bool = False,
               n_action: int = 1) -> Tuple[H2MG, float, Dict]:
        discrete_params, continuous_params = self.split_params(params)
        discrete_action, discrete_log_prob, discrete_info = self.discrete_sample(
            discrete_params, observation, rng=rng, deterministic=deterministic, n_action=n_action)
        continuous_observation = self.combine_obs_and_discrete_action(
            observation, discrete_action)
        assert(isinstance(continuous_observation, H2MG))
        continuous_action, continuous_log_prob, continuous_info = self.continuous_sample(
            continuous_params, continuous_observation, rng=rng, deterministic=deterministic, n_action=n_action)
        return discrete_action.combine(continuous_action), discrete_log_prob + continuous_log_prob, \
            discrete_info | continuous_info | {"continuous_log_prob": continuous_log_prob,
                                               "discrete_log_prob": discrete_log_prob}

    @partial(jit, static_argnums=(0, 4, 5))
    def vmap_sample(self, params: dict, observation: dict, rng, deterministic: bool = False, n_action: int = 1) -> float:
        return vmap(self.sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, observation, rng, deterministic,
                                                                                       n_action)
    def discrete_forward(self, params: Dict, observation: H2MG) -> H2MG:
        observation = manual_normalization(observation)  # TODO: remove this
        observation = self.normalizer(observation)
        logits = self.discrete_nn.apply(params, observation)
        return logits
    
    def discrete_log_prob(self, params, observation, action):
        discrete_action = action.extract_from_structure(
            self.discrete_action_space.structure)
        one_hot = self._action_to_one_hot(discrete_action)
        logits = self.discrete_forward(params, observation)
        return h2mg_categorical_logprob(one_hot, logits), logits

    def discrete_sample(self, params: Dict, observation: H2MG, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        logits = self.discrete_forward(params, observation)
        if n_action <= 1:
            one_hot = h2mg_categorical_sample(
                rng, logits, deterministic=deterministic)
            action = self._one_hot_to_action(one_hot)
            log_prob = h2mg_categorical_logprob(one_hot, logits)
        else:
            one_hot = [h2mg_categorical_sample(rng, logits, deterministic=deterministic) for rng in
                       jax.random.split(rng, n_action)]
            action = [self._one_hot_to_action(one_hot) for o in one_hot]
            log_prob = [h2mg_categorical_logprob(o, logits) for o in one_hot]
        info = h2mg.shallow_repr(logits.apply(
            lambda x: jnp.asarray(jnp.mean(x))))
        return action, log_prob, info

    @partial(jit, static_argnums=(0, 4, 5))
    def vmap_discrete_sample(self, params, observation: H2MG, rng, deterministic=False, n_action=1):
        return vmap(self.discrete_sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, observation, rng,
                                                                                                deterministic, n_action)

    def _one_hot_to_action(self, one_hot: H2MG):
        one_hot_multi_binary = one_hot.extract_from_structure(
            self.multi_binary_struct)
        one_hot_multi_discrete_up = one_hot.extract_from_structure(
            self.multi_discrete_up_struct)
        one_hot_multi_discrete_down = one_hot.extract_from_structure(
            self.multi_discrete_down_struct)
        action_multi_binary = one_hot_multi_binary
        action_multi_discrete = H2MG.from_structure(self.multi_discrete_struct)
        action_multi_discrete.flat_array = one_hot_multi_discrete_up.flat_array - \
            one_hot_multi_discrete_down.flat_array + 1
        return action_multi_binary.combine(action_multi_discrete)

    def _action_to_one_hot(self, action: H2MG):
        action_multi_binary = action.extract_from_structure(
            self.multi_binary_struct)
        action_multi_discrete_up = H2MG.from_structure(
            self.multi_discrete_up_struct)
        action_multi_discrete_down = H2MG.from_structure(
            self.multi_discrete_down_struct)
        action_multi_discrete = action.extract_from_structure(
            self.multi_discrete_struct)
        action_multi_discrete_up.flat_array = jnp.maximum(
            (action_multi_discrete.flat_array - 1), 0)
        action_multi_discrete_down.flat_array = jnp.maximum(
            (-action_multi_discrete.flat_array + 1), 0)
        one_hot_multi_binary = action_multi_binary
        return one_hot_multi_binary.combine(action_multi_discrete_up).combine(action_multi_discrete_down)

    def _build_postprocessor(self, h2mg_space: H2MGSpace):
        high = h2mg_space.continuous.high
        low = h2mg_space.continuous.low
        mu_0 = H2MG.from_structure(self.mu_structure)
        mu_0.flat_array = (high.flat_array + low.flat_array) / 2.
        log_sigma_0 = H2MG.from_structure(self.mu_structure)
        log_sigma_0.flat_array = jnp.log(
            (high.flat_array - low.flat_array) / self.box_to_sigma_ratio)
        return mu_0, log_sigma_0

    def _clip_log_sigma(self, log_sigma, eps=0.01):
        # soft clipping
        return jax.numpy.clip((1-eps)*log_sigma, a_min=jnp.log(self.clip_sigma)) + eps*log_sigma

    def _postprocess_distrib_params(self, distrib_params: H2MG):
        mu = distrib_params.extract_from_structure(self.mu_structure)
        log_sigma = distrib_params.extract_from_structure(
            self.log_sigma_structure)
        mu_norm = H2MG.from_structure(self.continuous_action_space.structure)
        mu_norm.flat_array = jnp.exp(
            self.log_sigma_0.flat_array) * mu.flat_array + self.mu_0.flat_array
        log_sigma_norm = H2MG.from_structure(
            self.continuous_action_space.structure)
        log_sigma_norm.flat_array = log_sigma.flat_array + self.log_sigma_0.flat_array
        if self.clip_sigma is not None and isinstance(self.clip_sigma, Number):
            log_sigma_norm.flat_array = self._clip_log_sigma(
                log_sigma_norm.flat_array)
        if self.cst_sigma is not None and isinstance(self.cst_sigma, Number):
            log_sigma_norm.flat_array = jnp.full_like(
                log_sigma.flat_array, jnp.log(self.cst_sigma))  # constant sigma
        return mu_norm, log_sigma_norm

    def continuous_log_prob(self, params, observation, action: H2MG):
        mu_norm, log_sigma_norm = self.continuous_forward(params, observation)
        continous_action = action.extract_from_structure(
            self.continuous_action_space.structure)
        return h2mg_normal_logprob(continous_action, mu_norm, log_sigma_norm), (mu_norm, log_sigma_norm)

    def continuous_forward(self, params: Dict, observation: H2MG) -> Tuple[H2MG, H2MG]:
        observation = manual_normalization(observation)  # TODO: remove this
        observation = self.normalizer(observation)
        distrib_params = self.continuous_nn.apply(params, observation)
        mu, log_sigma = self._postprocess_distrib_params(distrib_params)
        return mu, log_sigma

    def _continuous_sample(self, params, observation: H2MGSpace, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        mu_norm, log_sigma_norm = self.continuous_forward(params, observation)
        if n_action <= 1:
            action = h2mg_normal_sample(
                rng, mu_norm, log_sigma_norm, deterministic=deterministic)
            log_prob = h2mg_normal_logprob(action, mu_norm, log_sigma_norm)
        else:
            action = [h2mg_normal_sample(rng, mu_norm, log_sigma_norm, deterministic=deterministic)
                      for rng in jax.random.split(rng, n_action)]
            log_prob = [h2mg_normal_logprob(
                a, mu_norm, log_sigma_norm) for a in action]
        info = ContinuousPolicy.compute_info(mu_norm, log_sigma_norm)
        return action, log_prob, info, mu_norm, log_sigma_norm

    def continuous_sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        action, log_prob, info, _, _ = self._continuous_sample(
            params, observation, rng, deterministic, n_action)
        return action, log_prob, info

    @partial(jit, static_argnums=(0, 4, 5))
    def vmap_continuous_sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1):
        action, log_prob, _, mu, log_sigma = vmap(self._continuous_sample, in_axes=(
            None, 0, 0, None, None), out_axes=(0, 0, 0, 0, 0))(params, observation, rng, deterministic, n_action)
        info = ContinuousPolicy.compute_info(mu, log_sigma)
        batch_info = ContinuousPolicy.compute_batch_info(mu, log_sigma)
        return action, log_prob, info | batch_info

    def entropy(self, log_prob_info: Dict, batch=True):
        logits, (mu_norm, log_sigma_norm) = self.split_params(log_prob_info)
        if batch:
            return vmap_h2mg_categorical_entropy(logits)
        else:
            raise ValueError
