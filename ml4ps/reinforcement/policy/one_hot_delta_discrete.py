import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
import numpy as np
from ml4ps import Normalizer, h2mg
from ml4ps.reinforcement.policy.base import BasePolicy


class OneHotDeltaDiscrete(BasePolicy):
    def __init__(self, env=None, normalizer=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, **nn_args):
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.normalizer = normalizer or self.build_normalizer(env, normalizer_args)
        self.multi_discrete = env.action_space.multi_discrete.feature_dimension * 2
        self.multi_binary = env.action_space.multi_binary.feature_dimension * 1
        self.nn = ml4ps.neural_network.get(nn_type, {
            "feature_dimension": self.multi_binary.combine(self.multi_discrete), **nn_args})

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def _check_valid_action(self, action):
        pass

    def log_prob(self, params, observation, action):
        one_hot = self._action_to_one_hot(action)
        norm_observation = self.normalizer(observation)
        logits = self.nn.apply(params, norm_observation)
        return h2mg.categorical_logprob(one_hot, logits)

    def sample(self, params, observation, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        norm_observation = self.normalizer(observation)
        logits = self.nn.apply(params, norm_observation)
        if n_action <= 1:
            one_hot = h2mg.categorical(rng, logits, deterministic=deterministic)
            action = self._one_hot_to_action(one_hot)
            log_prob = h2mg.categorical_logprob(one_hot, logits)
        else:
            one_hot = [h2mg.categorical(rng, logits, deterministic=deterministic) for rng in
                jax.random.split(rng, n_action)]
            action = [self._one_hot_to_action(one_hot) for o in one_hot]
            log_prob = [h2mg.categorical_logprob(one_hot, logits) for o in one_hot]
        info = h2mg.shallow_repr(h2mg.map_to_features(lambda x: jnp.asarray(jnp.mean(x)), [logits]))
        return action, log_prob, info

    def _one_hot_to_action(self, one_hot):
        one_hot_multi_binary = one_hot.extract_like(self.multi_binary)
        one_hot_multi_discrete = one_hot.extract_like(self.multi_discrete)
        action_multi_binary = one_hot_multi_binary
        action_multi_discrete = one_hot_multi_discrete[:, 0] - one_hot_multi_discrete[:, 1] + 1
        return action_multi_binary.combine(action_multi_discrete)

    def _action_to_one_hot(self, action):
        action_multi_binary = action.extract_like(self.multi_binary)
        action_multi_discrete = action.extract_like(self.multi_discrete)
        pos = (action_multi_discrete - 1).maximum(0)
        neg = (-action_multi_discrete + 1).maximum(0)
        one_hot_multi_discrete = pos.stack(neg)
        one_hot_multi_binary = action_multi_binary
        return one_hot_multi_binary.combine(one_hot_multi_discrete)

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


