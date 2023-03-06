from typing import Dict, List, Tuple, Any, Sequence
from ml4ps.reinforcement.policy import get_policy, BasePolicy
from jax.random import PRNGKey, split
from jax import vmap, jit, value_and_grad
from ml4ps.reinforcement.environment import PSBaseEnv
import optax
from tqdm import tqdm
import numpy as np
from functools import partial
import pickle
import os
from flax.training import train_state
from collections import defaultdict

class ReinforceTrainState(train_state.TrainState):
    pass

def create_train_state(*, module, apply_fn, rng, learning_rate, init_obs):
    """Creates an initial `TrainState`."""
    params = module.init(rng, init_obs)
    tx = optax.adam(learning_rate=learning_rate)
    return ReinforceTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def remove_underscore_keys(d: Dict[str, Any]) -> Dict[str, Any]:
    keys_to_remove = []
    for k in d.keys():
        if k.startswith('_'):
            keys_to_remove.append(k)
    for k in keys_to_remove:
        del d[k]
    return d

class Reinforce():
    policy: BasePolicy
    env: PSBaseEnv
    def __init__(self, env: PSBaseEnv, seed=0, *, init_obs, val_env: PSBaseEnv=None, test_env: PSBaseEnv=None, policy_type: str=None, logger=None, validation_interval=100) -> None:
        super().__init__()
        self.policy = get_policy(policy_type, env, {})
        self.env = env
        self.seed = seed
        self.val_env = val_env or env
        self.test_env = test_env or env
        self.logger = logger
        self.validation_interval = validation_interval
        self.train_state = create_train_state(module=self.policy, apply_fn=self.vmap_sample, rng=PRNGKey(seed), learning_rate=1e-3, init_obs=init_obs)
    
    @property
    def hparams(self):
        return {"validation_interval": self.validation_interval, "seed": self.seed}
    
    @hparams.setter
    def hparams(self, value):
        self.validation_interval = value.get("validation_interval", self.validation_interval)
        self.seed = value.get("seed", self.seed)
        
    def train_step(self, state, batch, *, rng, batch_size):
        action, _, sample_info = state.apply_fn(state.params, batch, split(rng, batch_size))
        next_obs, rewards, done, _, step_info = self.env.step(action)
        value, grad = self.value_and_grad_fn(state.params, batch, action, rewards)
        state = state.apply_gradients(grads=grad)
        return state, next_obs, rewards, sample_info, step_info
    
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
        if self.val_env is not None:
            self.val_env.reset(seed=seed)
        for i in tqdm(range(n_iterations)):
            # Train step
            rng_key, subkey = split(rng_key)
            self.train_state, obs, rewards, sample_info, step_info = self.train_step(self.train_state, obs, rng=subkey, batch_size=batch_size)
            
            # Logging
            self.log_dicts(logger, i, "train_", sample_info, step_info, {"reward": rewards})

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
    

    def test(self, step, logger=None, seed=None, batch_size=None, test_env=None, rng_key=None):
        # [WIP]
        test_env = test_env or self.test_env
        old_cyclic_mode = test_env.cyclic_mode
        old_max_iter = test_env.max_iter
        test_env.cyclic_mode = True
        test_env.max_iter = 100
        obs = test_env.reset(seed=seed)
        while not test_env.empty():
            rng_key, subkey = split(rng_key)
            action, _, sample_info = self.vmap_sample_det(self.params, obs, split(subkey, batch_size))
            next_obs, rewards, done, _, step_info = test_env.step(action)
            self.log_dicts(logger, step, "test_", sample_info, step_info, {"reward": rewards})
        test_env.cyclic_mode = old_cyclic_mode
        test_env.max_iter = old_max_iter
    

    @partial(jit, static_argnums=(0,))
    def value_and_grad_fn(self, params, observations, actions, rewards):
        return value_and_grad(self.loss_fn)(params, observations, actions, rewards)
    
    def loss_fn(self, params, observations, actions, rewards):
        log_probs = self.vmap_log_prob(params, observations, action=actions)
        return - (log_probs * rewards).mean()
    
    @partial(jit, static_argnums=(0,))
    def vmap_sample(self, params, obs, seed) -> Tuple[Sequence[Any], Sequence[float], Dict[str, Any]]:
        return vmap(self.policy.sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0))(params, obs, seed, False, 1)
        
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
            res[prefix+"final_"+key] = np.nanmean([info[key] for info in not_none_infos])
        return res
    
    def log_final_info(self, logger, infos, step, prefix):
        res = self.get_final_info_dict(infos, prefix)
        logger.log_metrics_dict(res, step=step)

    def log(self, key, value, step):
        self.logger.log_metrics(key, value, step)

    def load(self, filename):
        # [WIP]
        pass
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.__dict__.update(data)

    def save(self, folder):
        # [WIP]
        pass
        self.policy.save(folder)
        self.env.save(folder)
        with open(os.path.join(folder, "hparams.pkl"), 'wb') as f:
            pickle.dump(self.hparams, f)
        with open(os.path.join(folder, "train_state.pkl"), 'wb') as f:
            pickle.dump(self.train_state, f)