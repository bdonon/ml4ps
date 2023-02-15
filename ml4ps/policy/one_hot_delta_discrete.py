from dataclasses import dataclass
from typing import Any, Dict, Tuple

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from gymnasium import spaces
from ml4ps import Normalizer, h2mg
from ml4ps.policy.base import BasePolicy

from .utils import (add_prefix, combine_feature_names, flatten_dict,
                    slice_with_prefix, space_to_feature_names, unflatten_like)


class OneHotDeltaDiscrete(BasePolicy):
    def __init__(self, env=None, normalizer=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, box_to_sigma_ratio=8, **nn_args):
        self.plus_prefix = "plus_"
        self.minus_prefix = "minus_"
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.action_space, self.observation_space = env.action_space, env.observation_space
        self.normalizer = normalizer or self.build_normalizer(env, normalizer_args)
        output_feature_names = self.build_output_feature_names(env.action_space)
        self.postprocessor =  self.build_postprocessor(env.action_space)
        self.nn = self.build_nn(nn_type, output_feature_names, **nn_args)

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def build_output_feature_names(self, action_space: spaces.Space) -> Dict:
        """Builds output feature names structure from power system space.
            The output feature correspond to the parameter of the continuous distribution.
        """
        feat_names = space_to_feature_names(action_space)
        plus_feat_names = add_prefix(feat_names, self.plus_prefix)
        minus_names = add_prefix(feat_names, self.minus_prefix)
        output_feature_names = combine_feature_names(plus_feat_names, minus_names)
        # TODO change the way stop is handled ?
        if "plus_stop" in h2mg.global_features(output_feature_names) and "minus_stop" in h2mg.global_features(output_feature_names):
            output_feature_names["global_features"].remove("plus_stop")
            output_feature_names["global_features"].remove("minus_stop")
            output_feature_names["global_features"].append("stop")
        return output_feature_names

    def build_nn(self, nn_type: str, output_feature_names: Dict, **kwargs):
        return ml4ps.neural_network.get(nn_type, {"output_feature_names": output_feature_names, **kwargs})

    def build_postprocessor(self, action_space: spaces.Space):
        """Builds postprocessor that transform nn output into the proper range via affine transformation.
        """
        @dataclass
        class PostProcessor():
            def __call__(self, distrib_params) -> Any:
                return distrib_params
        
        return PostProcessor()

    def _check_valid_action(self, action):
        pass
        # return h2mg.map_to_features(self._check_valid_action_feature, [action])

    def _check_valid_action_feature(self, action_feature):
        pass
        # if jnp.any((action_feature != 0) * (action_feature!= 1) * (action_feature!= 2)):
        #     raise ValueError

    def log_prob(self, params, observation, action):
        self._check_valid_action(action)
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        distrib_params = self.postprocessor(distrib_params)
        return self.categorical_log_prob(action, distrib_params)
    
    def categorical_log_prob(self, action, distrib_params):
        self._check_valid_action(action)
        action_onehot = self.action_to_onehot(action)
        flat_onehot = flatten_dict(action_onehot)
        flat_logits = flatten_dict(distrib_params)
        logits = jax.nn.log_softmax(flat_logits)
        logits_selected= logits*jax.lax.stop_gradient(flat_onehot)
        return jnp.sum(logits_selected)

    def sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1) -> Tuple[spaces.Space, float]:
        """Sample an action and return it together with the corresponding log probability."""
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        distrib_params = self.postprocessor(distrib_params)
        if n_action <= 1:
            action = self.sample_from_params(rng, distrib_params, deterministic=deterministic)
            log_prob = self.categorical_log_prob(action, distrib_params)
        else:
            action = [self.sample_from_params(rng, distrib_params, deterministic=deterministic) for rng, _ in zip(jax.random.split(rng), range(n_action))]
            log_prob = [self.categorical_log_prob(a, distrib_params) for a in action]
        info = h2mg.shallow_repr(h2mg.map_to_features(lambda x: jnp.asarray(jnp.mean(x)), [distrib_params]))
        return action, log_prob, info
    
    def split_params(self, out_dict):
        minus = slice_with_prefix(out_dict, self.minus_prefix)
        minus["global_features"]["stop"] = out_dict["global_features"]["stop"]
        plus = slice_with_prefix(out_dict, self.plus_prefix)
        plus["global_features"]["stop"] = out_dict["global_features"]["stop"]
        return minus, plus
    
    def sample_from_params(self, rng, distrib_params: Dict, deterministic=False) -> Dict:
        """Sample an action from the parameter of the continuous distribution."""
        flat_action = flatten_dict(distrib_params)
        if deterministic:
            idx = jnp.argmax(flat_action)
        else:
            idx = jax.random.categorical(key=rng, logits=flat_action)
        flat_res_action = jnp.zeros_like(flat_action)
        flat_res_action = flat_res_action.at[idx].set(1)
        res_action = unflatten_like(flat_res_action, distrib_params)
        res_action = self.onehot_to_action(res_action)
        return res_action
    
    def action_to_onehot(self, action):
        res = h2mg.empty_h2mg()
        for local_key, obj_name, feat_name, value in h2mg.local_features_iterator(action):
            res[local_key][obj_name][self.minus_prefix+feat_name] = jnp.float32(value == 0)
            res[local_key][obj_name][self.plus_prefix+feat_name] = jnp.float32(value == 2)
        for global_key, feat_name, value in h2mg.global_features_iterator(action):
            res[global_key][self.minus_prefix+feat_name] = jnp.float32(value == 0)
            res[global_key][self.plus_prefix+feat_name] = jnp.float32(value == 2)
        del res["global_features"][self.minus_prefix+"stop"]
        del res["global_features"][self.plus_prefix+"stop"]
        res["global_features"]["stop"] = action["global_features"]["stop"]
        return res
    
    def onehot_to_action(self, onehot):
        minus, plus = self.split_params(onehot)
        res = h2mg.map_to_features(self.onehot_to_action_feature, [minus, plus])
        res["global_features"]["stop"] = minus["global_features"]["stop"] * plus["global_features"]["stop"]
        return res
    
    def onehot_to_action_feature(self, minus, plus):
        res = jnp.ones_like(minus)
        res = jnp.where(minus == 1, 0, res)
        res = jnp.where(plus == 1, 2, res)
        return res
    
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

    
