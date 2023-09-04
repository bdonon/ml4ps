from functools import partial
from typing import Callable, Dict

import gymnasium as gym
import jax
import jax.numpy as jnp
import ml4ps
from jax import jit, vmap
from ml4ps import h2mg
from ml4ps.h2mg import (H2MG, H2MGStructure, h2mg_categorical_logprob,
                        h2mg_categorical_sample, vmap_h2mg_categorical_entropy)
from ml4ps.reinforcement.policy.base import BasePolicy


def one_hot_to_action(one_hot: H2MG, multi_binary_struct: H2MGStructure, multi_discrete_up_struct: H2MGStructure,
                      multi_discrete_down_struct: H2MGStructure, multi_discrete_struct: H2MGStructure) -> H2MG:
    one_hot_multi_binary = one_hot.extract_from_structure(multi_binary_struct)
    one_hot_multi_discrete_up = one_hot.extract_from_structure(
        multi_discrete_up_struct)
    one_hot_multi_discrete_down = one_hot.extract_from_structure(
        multi_discrete_down_struct)
    action_multi_binary = one_hot_multi_binary
    action_multi_discrete = H2MG.from_structure(multi_discrete_struct)
    action_multi_discrete.flat_array = one_hot_multi_discrete_up.flat_array - \
        one_hot_multi_discrete_down.flat_array + 1
    return action_multi_binary.combine(action_multi_discrete)


def action_to_one_hot(action: H2MG, multi_binary_struct: H2MGStructure, multi_discrete_up_struct: H2MGStructure,
                      multi_discrete_down_struct: H2MGStructure, multi_discrete_struct: H2MGStructure) -> H2MG:
    action_multi_binary = action.extract_from_structure(multi_binary_struct)
    action_multi_discrete_up = H2MG.from_structure(multi_discrete_up_struct)
    action_multi_discrete_down = H2MG.from_structure(
        multi_discrete_down_struct)
    action_multi_discrete = action.extract_from_structure(
        multi_discrete_struct)
    action_multi_discrete_up.flat_array = jnp.maximum(
        (action_multi_discrete.flat_array - 1), 0)
    action_multi_discrete_down.flat_array = jnp.maximum(
        (-action_multi_discrete.flat_array + 1), 0)
    one_hot_multi_binary = action_multi_binary
    return one_hot_multi_binary.combine(action_multi_discrete_up).combine(action_multi_discrete_down)


def discrete_log_prob(params: Dict, observation: H2MG, action: H2MG, normalizer: Callable, nn):
    one_hot = action_to_one_hot(action)
    norm_observation = normalizer(observation)
    logits = nn.apply(params, norm_observation)
    return h2mg_categorical_logprob(one_hot, logits), logits


class OneHotDeltaDiscrete(BasePolicy):
    # TODO cst_sigma ? remove
    def __init__(self, env: gym.Env | gym.vector.VectorEnv, normalizer=None, normalizer_args=None, nn_type="h2mgnode",
                 cst_sigma=None, nn_args={}):
        self.nn_args = nn_args
        self.normalizer = normalizer or self._build_normalizer(
            env, normalizer_args)
        if isinstance(env, gym.vector.VectorEnv):
            multi_discrete: h2mg.H2MGSpace = env.single_action_space.multi_discrete
            self.multi_binary_struct: h2mg.H2MGStructure = env.single_action_space.multi_binary.structure
        else:
            multi_discrete: h2mg.H2MGSpace = env.action_space.multi_discrete
            self.multi_binary_struct: h2mg.H2MGStructure = env.action_space.multi_binary.structure
        self.multi_discrete_struct = multi_discrete.structure
        self.multi_discrete_up_struct = multi_discrete.add_suffix(
            "_up").structure
        self.multi_discrete_down_struct = multi_discrete.add_suffix(
            "_down").structure
        output_structure = self.multi_binary_struct.combine(
            self.multi_discrete_up_struct).combine(self.multi_discrete_down_struct)
        # print(output_structure)
        self.nn = ml4ps.neural_network.get(nn_type,
                                           output_structure=output_structure,
                                           **nn_args)

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def _check_valid_action(self, action):
        pass

    def log_prob(self, params, observation, action):
        one_hot = self._action_to_one_hot(action)
        norm_observation = self.normalizer(observation)
        logits = self.nn.apply(params, norm_observation)
        return h2mg_categorical_logprob(one_hot, logits), logits

    def sample(self, params: Dict, observation: H2MG, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        norm_observation = self.normalizer(observation)
        logits: H2MG = self.nn.apply(params, norm_observation)
        if n_action <= 1:
            one_hot = h2mg_categorical_sample(
                rng, logits, deterministic=deterministic)
            action = self._one_hot_to_action(one_hot)
            log_prob = h2mg_categorical_logprob(one_hot, logits)
            # assert(log_prob == self.log_prob(params, observation, action))
        else:
            one_hot = [h2mg_categorical_sample(rng, logits, deterministic=deterministic) for rng in
                       jax.random.split(rng, n_action)]
            action = [self._one_hot_to_action(one_hot) for o in one_hot]
            log_prob = [h2mg_categorical_logprob(o, logits) for o in one_hot]
        info = h2mg.shallow_repr(logits.apply(
            lambda x: jnp.asarray(jnp.mean(x))))
        return action, log_prob, info

    @partial(jit, static_argnums=(0, 4, 5))
    def vmap_sample(self, params, observation: H2MG, rng, deterministic=False, n_action=1):
        return vmap(self.sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, observation, rng, deterministic,
                                                                                       n_action)

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

    def entropy(self, log_prob_info: Dict, batch=True):
        logits = log_prob_info
        if batch:
            return vmap_h2mg_categorical_entropy(logits)
        else:
            raise ValueError
