from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from gymnasium import spaces
from ml4ps import Normalizer, h2mg
from ml4ps.policy.base import BasePolicy


def add_prefix(x, prefix):
    return transform_feature_names(x, lambda feat_name: prefix+feat_name)


def remove_prefix(x, prefix):
    return transform_feature_names(x, lambda feat_name: feat_name.removeprefix(prefix))


def tr_feat(feat_names, fn):
    if isinstance(feat_names, list):
        return list(map(fn, feat_names))
    elif isinstance(feat_names, dict):
        return {fn(feat): value for feat, value in feat_names.items()}


def transform_feature_names(_x, fn: Callable):
    x = _x.copy()
    if "local_features" in x:
        x |= {"local_features": {obj_name: tr_feat(
            obj, fn) for obj_name, obj in x["local_features"].items()}}
    if "global_features" in x:
        x |= {"global_features": tr_feat(x["global_features"], fn)}
    return x


def slice_with_prefix(_x, prefix):
    x = _x.copy()
    if "local_features" in x:
        x |= {"local_features": {obj_name: {feat.removeprefix(prefix): value for feat, value in obj.items(
        ) if feat.startswith(prefix)} for obj_name, obj in x["local_features"].items()}}
    if "global_features" in x:
        x |= {"global_features": {feat.removeprefix(
            prefix): value for feat, value in x["global_features"] if feat.startswith(prefix)}}
    return x


def combine_space(a, b):
    x = h2mg.empty_h2mg()
    for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(a):
        x[local_key][obj_name][feat_name] = value
    for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(b):
        x[local_key][obj_name][feat_name] = value

    for global_key,  feat_name, value in h2mg.global_features_iterator(a):
        x[global_key][feat_name] = value
    for global_key,  feat_name, value in h2mg.global_features_iterator(b):
        x[global_key][feat_name] = value

    for local_key, obj_name, addr_name, value in h2mg.local_addresses_iterator(a):
        x[local_key][obj_name][addr_name] = value

    for all_addr_key, value in h2mg.all_addresses_iterator(a):
        x[all_addr_key][value] = value
    return x


def combine_feature_names(feat_a, feat_b):
    new_feat_a = defaultdict(lambda: defaultdict(list))
    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_a):
        new_feat_a[local_key][obj_name].append(feat_name)
    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_b):
        new_feat_a[local_key][obj_name].append(feat_name)

    for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(feat_b):
        new_feat_a[local_key][obj_name] = list(
            set(new_feat_a[local_key][obj_name]))

    new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value] = list()
    for global_key, feat_name in h2mg.global_feature_names_iterator(feat_a):
        new_feat_a[global_key].append(feat_name)
    for global_key, feat_name in h2mg.global_feature_names_iterator(feat_b):
        new_feat_a[global_key].append(feat_name)
    new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value] = list(
        set(new_feat_a[h2mg.H2MGCategories.GLOBAL_FEATURES.value]))
    return new_feat_a


def space_to_feature_names(space: spaces.Space):
    feat_names = {}
    if "local_addresses" in list(space.keys()):
        feat_names |= {"local_addresses": {
            k: list(v) for k, v in space["local_addresses"].items()}}
    if "local_features" in list(space.keys()):
        feat_names |= {"local_features": {
            k: list(v) for k, v in space["local_features"].items()}}
    if "global_features" in list(space.keys()):
        feat_names |= {"global_features": {
            list(k) for k, _ in space["global_features"].items()}}
    return feat_names

# TODO handle nan in observation ?


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
        self.mu_prefix = "mu_"
        self.log_sigma_prefix = "log_sigma_"
        self.box_to_sigma_ratio = box_to_sigma_ratio
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.action_space, self.observation_space = env.action_space, env.observation_space
        self.normalizer = normalizer or self.build_normalizer(env, normalizer_args)
        output_feature_names = self.build_output_feature_names(env.action_space)
        self.postprocessor =  self.build_postprocessor(env.action_space)
        self.nn = self.build_nn(nn_type, output_feature_names,
                                local_dynamics_hidden_size=[16],
                                global_dynamics_hidden_size=[16],
                                local_decoder_hidden_size=[16],
                                global_decoder_hidden_size=[16],
                                local_latent_dimension=4,
                                global_latent_dimension=4,
                                stepsize_controller_name="ConstantStepSize",
                                stepsize_controller_kwargs={})
        
    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def build_output_feature_names(self, action_space: spaces.Space) -> Dict:
        """Builds output feature names structure from power system space.
            The output feature correspond to the parameter of the continuous distribution.
        """
        feat_names = space_to_feature_names(action_space)
        log_sigma_feat_names = add_prefix(feat_names, self.log_sigma_prefix)
        mu_feature_names = add_prefix(feat_names, self.mu_prefix)
        output_feature_names = combine_feature_names(log_sigma_feat_names, mu_feature_names)
        return output_feature_names

    def build_nn(self, nn_type: str, output_feature_names: Dict, **kwargs):
        return ml4ps.neural_network.get(nn_type, {"output_feature_names":output_feature_names, **kwargs})

    def build_postprocessor(self, action_space: spaces.Space):
        """Builds postprocessor that transform nn output into the proper range via affine transformation.
        """
        post_process_h2mg = h2mg.empty_h2mg()
        for local_key, obj_name, feat_name in h2mg.local_feature_names_iterator(space_to_feature_names(action_space)):
            high = action_space[local_key][obj_name][feat_name].high
            low = action_space[local_key][obj_name][feat_name].low
            sigma = jnp.mean((high-low)/self.box_to_sigma_ratio)
            post_process_h2mg[local_key][obj_name][self.log_sigma_prefix + feat_name] = lambda x: x+jnp.log(sigma)
            post_process_h2mg[local_key][obj_name][self.mu_prefix + feat_name] = lambda x: x*sigma+jnp.mean(low + (high-low)/2)
        for global_key, feat_name in h2mg.global_feature_names_iterator(space_to_feature_names(action_space)):
            high = action_space[global_key][feat_name].high
            low = action_space[global_key][feat_name].low
            sigma = jnp.mean((high-low)/self.box_to_sigma_ratio)
            post_process_h2mg[global_key][self.log_sigma_prefix + feat_name] = lambda x: x+jnp.log(sigma)
            post_process_h2mg[global_key][self.mu_prefix + feat_name] = lambda x: x*sigma+jnp.mean(low + (high-low)/2)


        @dataclass
        class PostProcessor():
            post_process_h2mg: Dict
            def __call__(self, distrib_params) -> Any:
                return h2mg.map_to_features(lambda target, fn: fn(target), [distrib_params, self.post_process_h2mg])
        
        return PostProcessor(post_process_h2mg)

    def _check_valid_action(self, action):
        # TODO
        pass

    def log_prob(self, params, observation, action):
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        distrib_params = self.postprocessor(distrib_params)
        return self.normal_log_prob(action, distrib_params)

    def normal_log_prob(self, action: Dict, distrib_params: Dict) -> float:
        """Return the log probability of an action"""
        self._check_valid_action(action)
        mu, log_sigma = self.split_params(distrib_params)
        log_probs = h2mg.map_to_features(self.feature_log_prob, [action, mu, log_sigma])
        return sum(h2mg.features_iterator(log_probs))

    def feature_log_prob(self, action, mu, log_sigma):
        return jnp.nansum(- log_sigma - 0.5 * jnp.exp(-2 * log_sigma) * (action - mu)**2)

    def sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1) -> Tuple[spaces.Space, float]:
        # n_action = 1, no list, n_action > 1 list
        if n_action > 1:
            raise NotImplementedError
        """Sample an action and return it together with the corresponding log probability."""
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        distrib_params = self.postprocessor(distrib_params)
        if n_action <= 1:
            action = self.sample_from_params(rng, distrib_params, deterministic=deterministic)
            log_prob = self.normal_log_prob(action, distrib_params)
        else:
            action = [self.sample_from_params(rng, distrib_params, deterministic=deterministic) for _ in range(n_action)]
            log_prob = [self.normal_log_prob(a, distrib_params) for a in action]
        info= {"info": 0}
        return action, log_prob, info

    def split_params(self, out_dict):
        mu = slice_with_prefix(out_dict, self.mu_prefix)
        log_sigma = slice_with_prefix(out_dict, self.log_sigma_prefix)
        return mu, log_sigma

    def sample_from_params(self, rng, distrib_params: Dict, deterministic=False) -> Dict:
        """Sample an action from the parameter of the continuous distribution."""
        mu, log_sigma = self.split_params(distrib_params)
        if deterministic:
            return mu
        return h2mg.map_to_features(lambda mu, log_sigma: mu + jax.random.normal(key=rng, shape=log_sigma.shape) * jnp.exp(log_sigma), [mu, log_sigma])

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
            return Normalizer(backend=backend, data_dir=data_dir, **normalizer_args) # TODO kwargs.get("normalizer_args", {})
