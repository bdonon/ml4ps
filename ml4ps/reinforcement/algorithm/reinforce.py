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
    def __init__(self, env: PSBaseEnv, seed=0, val_env: PSBaseEnv=None, test_env: PSBaseEnv=None, policy_type: str=None, logger=None, validation_interval=100) -> None:
        super().__init__()
        self.policy = get_policy(policy_type, env, {})
        self.env = env
        self.seed = seed
        self.val_env = val_env or env
        self.test_env = test_env or env
        self.logger = logger
        self.validation_interval = validation_interval
    
    @property
    def hparams(self):
        return {"validation_interval": self.validation_interval, "seed": self.seed}
    
    @hparams.setter
    def hparams(self, value):
        self.validation_interval = value.get("validation_interval", self.validation_interval)
        self.seed = value.get("seed", self.seed)
    
    @property
    def train_state(self):
        pass
    
    @hparams.setter
    def train_state(self, value):
        pass
    
    def init(self, seed, observation):
        self.params = self.policy.init(seed, obs=observation)
        self.optimizer, self.optimizer_state = self.configure_optimizers(self.params)

    
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
            action, _, sample_info = self.vmap_sample(self.params, obs, split(subkey, batch_size))
            next_obs, rewards, done, _, step_info = self.env.step(action)
            value, grad = self.value_and_grad_fn(self.params, obs, action, rewards)
            obs = next_obs
            updates, self.optimizer_state = self.optimizer.update(grad, self.optimizer_state)
            self.params = optax.apply_updates(self.params, updates)
            
            # Logging
            self.log_dicts(logger, i, "train_", sample_info, step_info, {"reward": rewards})

            # Val step
            if i % validation_interval==0 and self.val_env is not None:
                self.val_env.reset()
                for _ in range(10):
                    rng_key_val, subkey_val = split(rng_key_val)
                    action, _, sample_info = self.vmap_sample_det(self.params, obs, split(subkey_val, batch_size))
                    next_obs, rewards, done, _, step_info = self.val_env.step(action)
                    self.log_dicts(logger, i, "val_", sample_info, step_info, {"reward": rewards})

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
            
    def configure_optimizers(self, params):
        optimizer = optax.adam(learning_rate=1e-3)
        optimizer_state = optimizer.init(params)
        return optimizer, optimizer_state


    def log_dicts(self, logger, step, prefix, *dicts):
        for d in dicts:
            d = remove_underscore_keys(d)
            d = {prefix + k: v for (k, v) in d.items()}
            d = self.handle_vector_env(logger, d, step, prefix)
            logger.log_metrics_dict(d, step)
    
    def handle_vector_env(self, logger, d, step, prefix):
        keys_to_remove = []
        for k, v  in d.items():
            if k.endswith("final_observation"):
                keys_to_remove.append(k)
            elif k.endswith("final_info"):
                keys_to_remove.append(k)
                self.log_final_info(logger, v, step, prefix)
        for k in keys_to_remove:
            del d[k]
        return d
    
    def log_final_info(self, logger, infos, step, prefix):
        res = {}
        for key in infos[0]:
            res[prefix+"final_"+key] = np.nanmean([info[key] for info in infos])
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
            pickle.dump(self.hparams, f)