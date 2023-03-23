import os
import pickle
from collections import defaultdict
from copy import deepcopy
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
from gymnasium.vector.utils.spaces import iterate
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.random import PRNGKey, split
from jax.tree_util import tree_leaves
from ml4ps.neural_network import get as get_neural_network
from ml4ps.reinforcement.test_policy import test_policy
from ml4ps.reinforcement.environment import PSBaseEnv
from ml4ps.reinforcement.policy import BasePolicy, get_policy
from tqdm import tqdm


def get_states(env: gym.vector.VectorEnv) -> Any:
    return env.get_attr('state')

def set_states(env: gym.vector.VectorEnv, states) -> Any:
    return env.set_attr('state', states)




class ReinforceTrainState(train_state.TrainState):
    pass


def create_train_state(*, env, module, apply_fn, rng, learning_rate):
    """Creates an initial `TrainState`."""
    batch_obs, _ = env.reset()
    single_obs = next(iterate(env.observation_space, batch_obs))
    params = module.init(rng, single_obs)
    tx = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(learning_rate=learning_rate))
    return ReinforceTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def remove_underscore_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    keys_to_remove = []
    for k in d.keys():
        if k.startswith('_'):
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del d[k]
    return d

def count_params(params):
    return sum(x.size for x in tree_leaves(params))

class Reinforce:
    policy: BasePolicy
    env: PSBaseEnv
    def __init__(self, env: PSBaseEnv, seed=0, *, val_env: PSBaseEnv, test_env: PSBaseEnv, policy_type: str=None, logger=None, validation_interval=100, baseline=None, nn_args={}, clip_norm=0.1, learning_rate=0.0003) -> None:
        super().__init__()
        self.policy = get_policy(policy_type, env, nn_args)
        self.env = env
        self.seed = seed
        self.baseline = baseline
        self.val_env = val_env
        self.test_env = test_env
        self.logger = logger
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate
        self.validation_interval = validation_interval
        self.train_state = create_train_state(env=env, module=self.policy, apply_fn=self.vmap_sample, rng=PRNGKey(seed), learning_rate=learning_rate)
        self.init_baseline()
    
    @property
    def hparams(self):
        return {"validation_interval": self.validation_interval, "seed": self.seed, "baseline": self.baseline, "clip_norm": self.clip_norm, "learning_rate": self.learning_rate}
    
    @hparams.setter
    def hparams(self, value):
        self.validation_interval = value.get("validation_interval", self.validation_interval)
        self.seed = value.get("seed", self.seed)
    
    def init_baseline(self):
        if self.baseline is None:
            return
        if self.baseline != "model":
            return

    def compute_baseline(self, state, batch, rng, batch_size):
        if self.baseline is None:
            raise ValueError("Baseline must be specified")
        elif self.baseline == "mean" or self.baseline == "median":
            n_action = 5
            rewards = []
            for _ in range(n_action):
                baseline_action, _, _ = self.vmap_sample(state.params, batch, split(rng, batch_size))
                states = deepcopy(get_states(self.env))
                _, baseline_reward, _, _, _ = self.env.step(baseline_action)
                rewards.append(baseline_reward)
                set_states(self.env, states)
            rewards = jnp.stack(rewards, axis=0)
            baseline_rewards = jnp.mean(jnp.array(rewards), axis=0) if self.baseline == "mean" else jnp.median(jnp.array(rewards), axis=0)
        elif self.baseline == "deterministic":
            baseline_action, _, _ = self.vmap_sample_det(state.params, batch, split(rng, batch_size))
            states = deepcopy(get_states(self.env))
            _, baseline_rewards, _, _, _ = self.env.step(baseline_action)
            set_states(self.env, states)
        else:
            raise ValueError(f"Unknown baseline: {self.baseline}")
        return baseline_rewards

        
    def train_step(self, state, batch, *, rng, batch_size):
        if self.baseline is not None:
            baseline = self.compute_baseline(state, batch, rng, batch_size)
        else:
            baseline = 0
        action, log_probs, sample_info = state.apply_fn(state.params, batch, split(rng, batch_size))
        next_obs, rewards, done, _, step_info = self.env.step(action)
        value, grad = self.value_and_grad_fn(state.params, batch, action, rewards, baseline)
        state = state.apply_gradients(grads=grad)
        other_info = {"grad_norm": optax._src.linear_algebra.global_norm(grad),
                     "loss_value": value,
                     "log_probs": jnp.mean(log_probs)}
        return state, next_obs, rewards, sample_info, step_info, other_info
    
    def validation_step(self, state, batch, *, rng, batch_size):
        action, _, sample_info = self.vmap_sample_det(state.params, batch, split(rng, batch_size))
        next_obs, rewards, done, _, step_info = self.val_env.step(action)
        return sample_info, step_info, rewards

    def learn(self, logger=None, seed=None, batch_size=None, n_iterations=1000, validation_interval=None):
        validation_interval = validation_interval or self.validation_interval
        logger = logger or self.logger
        seed = seed or self.seed
        
        rng_key = PRNGKey(seed)
        rng_key, rng_key_val = split(rng_key)
        obs, _ = self.env.reset(seed=seed)
        self.val_env.reset(seed=seed)
        for i in tqdm(range(n_iterations)):
            # Train step
            rng_key, subkey = split(rng_key)
            self.train_state, obs, rewards, sample_info, step_info, other_info = self.train_step(self.train_state, obs, rng=subkey, batch_size=batch_size)
            
            # Logging
            self.log_dicts(logger, i, "train_", sample_info, step_info, {"reward": rewards}, other_info)

            # Val step
            if i % validation_interval==0 and self.val_env is not None:
                self.val_env.reset()
                val_metrics = defaultdict(list)
                for _ in range(10):
                    rng_key_val, subkey_val = split(rng_key_val)
                    sample_info, step_info, rewards = self.validation_step(self.train_state, obs, rng=subkey_val, batch_size=batch_size)
                    sample_info, step_info = self.process_dict(sample_info, "val_") ,self.process_dict(step_info, "val_")
                    for k, v in sample_info.items():
                        val_metrics[k].append(v)
                    for k, v in step_info.items():
                        val_metrics[k].append(v)
                    val_metrics["val_reward"].append(np.mean(rewards))
                val_metrics = self.dict_mean(val_metrics)
                logger.log_metrics_dict(val_metrics, i)
    

    def test(self, test_env, res_dir, test_name=None):
        test_name = test_name or "test"
        test_dir = os.path.join(res_dir, test_name)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        return test_policy(test_env, self.policy, self.train_state.params, seed=self.seed, output_dir=test_dir)
    

    @partial(jit, static_argnums=(0,))
    def value_and_grad_fn(self, params, observations, actions, rewards, baseline=0):
        return value_and_grad(self.loss_fn)(params, observations, actions, rewards, baseline)
    
    def loss_fn(self, params, observations, actions, rewards, baseline=0):
        log_probs = self.vmap_log_prob(params, observations, action=actions)
        # for action in actions:
        #     log_prob = ...
        # log_probs = stack
        return - (log_probs * (rewards -  baseline)).mean()
    
    @partial(jit, static_argnums=(0,))
    def vmap_sample(self, params, obs, seed) -> Tuple[Sequence[Any], Sequence[float], Dict[str, Any]]:
        return vmap(self.policy.sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, obs, seed, False, 1)
    
    @partial(jit, static_argnums=(0,4))
    def vmap_sample_mupltiple(self, params, obs, seed, n_actions) -> Tuple[Sequence[Any], Sequence[float], Dict[str, Any]]:
        return vmap(self.policy.sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, obs, seed, False, n_actions)
        
    @partial(jit, static_argnums=(0,))
    def vmap_sample_det(self, params, obs, seed) -> Tuple[Sequence[Any], Sequence[float], Dict[str, Any]]:
        return vmap(self.policy.sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, obs, seed, True, 1)

    def vmap_log_prob(self, params, obs, action):
        return vmap(self.policy.log_prob, in_axes=(None, 0, 0), out_axes=0)(params, obs, action)

    def log_dicts(self, logger, step, prefix, *dicts):
        for d in dicts:
            d = self.process_dict(d, prefix)
            logger.log_metrics_dict(d, step)
    
    def process_dict(self, d: Dict[str, Any], prefix) -> Dict[str, Any]:
        d = remove_underscore_keys(d)
        d = {prefix + k: v for (k, v) in d.items()}
        d = self.handle_vector_env( d, prefix)
        return  self.dict_mean(d)
    
    def dict_mean(self, d):
        return {k: np.nanmean(v) for (k, v) in d.items()}
    
    def handle_vector_env(self, d, prefix):
        keys_to_remove = []
        final_info = {}
        for k, v  in d.items():
            if k.endswith("final_observation"):
                keys_to_remove.append(k)
            elif k.endswith("final_info"):
                keys_to_remove.append(k)
                final_info = self.get_final_info_dict(v, prefix)
        for k in keys_to_remove:
            del d[k]
        return {**d, **final_info}
    
    def get_final_info_dict(self, infos, prefix):
        res = {}
        not_none_infos = [info for info in infos if info is not None]
        for key in not_none_infos[0]:
            try:
                res[prefix+"final_"+key] = np.nanmean([info[key] for info in not_none_infos])
            except:
                continue
        return res
    
    def log_final_info(self, logger, infos, step, prefix):
        res = self.get_final_info_dict(infos, prefix)
        logger.log_metrics_dict(res, step=step)

    def log(self, key, value, step):
        self.logger.log_metrics(key, value, step)
    
    def _policy_filename(self, folder):
        return os.path.join(folder, "policy.pkl")
    
    def _hparams_filename(self, folder):
        return os.path.join(folder, "hparams.pkl")
    
    def _train_state_filename(self, folder):
        return os.path.join(folder, "train_state.pkl")


    def load(self, folder):
        self.env = PSBaseEnv.load(self._policy_filename(folder))
        with open(self._hparams_filename(folder), 'rb') as f:
            self.hparams = pickle.load(f)
        with open(self._train_state_filename(folder), 'rb') as f:
            self.train_state = pickle.load(f)

    def save(self, folder):
        self.env.save(self._policy_filename(folder))
        with open(os.path.join(folder, self._hparams_filename(folder)), 'wb') as f:
            pickle.dump(self.hparams, f)
        with open(os.path.join(folder, self._train_state_filename(folder)), 'wb') as f:
            pickle.dump(self.train_state, f)
