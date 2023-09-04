import json
import os
import pickle
from functools import partial
from time import time
from typing import Any, Dict, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
import optax
from flax.training import train_state
import jax
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.random import PRNGKey, split
from jax.tree_util import tree_leaves
from ml4ps.h2mg import H2MG, collate_h2mgs
from ml4ps.logger import BaseLogger
from ml4ps.reinforcement.environment import PSBaseEnv
from ml4ps.reinforcement.policy import BasePolicy, get_policy
from ml4ps.reinforcement.test_policy import eval_reward, test_policy
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import nullcontext

from .algorithm import Algorithm


def jax_device_context():
    try:
        ctx = jax.default_device(jax.devices('gpu')[0])
    except:
        ctx = nullcontext()
    
    return ctx

# class ReinforceTrainState(train_state.TrainState):
#     pass

@dataclass
class ReinforceTrainState:
    step: int
    apply_fn: Any
    params: Any
    tx: Any
    opt_state: Any
    
    def apply_gradients(self, *, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
        step=self.step + 1,
        params=new_params,
        opt_state=new_opt_state,
        **kwargs,
    )

    @classmethod
    def create(cls, *, apply_fn, params, tx, **kwargs):
        """Creates a new instance with `step=0` and initialized `opt_state`."""
        opt_state = tx.init(params)
        return cls(
            step=0,
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )
    
    def replace(self, *, step, params, opt_state, **kwargs) :
        self.step = step
        self.params = params
        self.opt_state = opt_state
        return self

def create_train_state(*, single_obs, module, apply_fn, rng, learning_rate,
                       clip_norm=0.1, lr_schedule_kwargs=None, every_k_schedule=None) -> ReinforceTrainState:
    """Creates an initial `TrainState`."""
    params = module.init(rng, single_obs)
    # TODO: clip_by_values ?
    if lr_schedule_kwargs is not None:
        lr_schedule = optax.exponential_decay(**lr_schedule_kwargs)
    else:
        lr_schedule = learning_rate
    tx = optax.chain(optax.clip_by_global_norm(clip_norm),
                     optax.adam(learning_rate=lr_schedule))
    if every_k_schedule is not None and every_k_schedule > 1:
            tx = optax.MultiSteps(tx, every_k_schedule=every_k_schedule)  # Accumulate gradient for k batches
    return ReinforceTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def count_params(params):
    return sum(x.size for x in tree_leaves(params))


class Reinforce(Algorithm):
    policy: BasePolicy
    env: PSBaseEnv
    logger: BaseLogger
    train_state: ReinforceTrainState
    best_params_name :str = "best_params.pkl"
    last_params_name :str = "last_params.pkl"
    best_params_info_name: str = 'best_params_info.json'

    def __init__(self, *, env: PSBaseEnv, seed=0, val_env: PSBaseEnv, test_env: PSBaseEnv, run_dir,
                max_steps, policy_type: str, logger=None, validation_interval=100, baseline=None,
                policy_args={}, nn_args={}, clip_norm, learning_rate, lr_schedule_kwargs=None, n_actions=5,
                baseline_learning_rate=None, baseline_nn_type=None, baseline_nn_args=None, init_cost=None,
                nn_baseline_steps=None, normalize_baseline_rewards=False, gradient_steps=1, update_cst_sigma=None,
                update_cst_sigma_kwargs={}, alpha_entropy=None, alpha_std=None, mixed_std=False, exp_decay_std=None,
                clip_rewards=None, reward_normalization="normal", every_k_schedule=None) -> 'Reinforce':
        self.best_params_path = os.path.join(run_dir, self.best_params_name)
        self.last_params_path = os.path.join(run_dir, self.last_params_name)
        self.policy_type = policy_type
        self.max_steps = max_steps
        self.policy = get_policy(
            policy_type, env, **policy_args, nn_args=nn_args)
        self.env = env
        self.seed = seed
        self.baseline = baseline
        self.val_env = val_env
        self.test_env = test_env
        self.logger = logger
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate
        self.every_k_schedule = every_k_schedule
        self.lr_schedule_kwargs = lr_schedule_kwargs
        self.baseline_learning_rate = baseline_learning_rate
        self.baseline_nn_type = baseline_nn_type
        self.baseline_nn_args = baseline_nn_args
        self.nn_baseline_steps = nn_baseline_steps
        self.validation_interval = validation_interval
        self.n_actions = n_actions
        self.run_dir = run_dir
        self.init()
        self.init_cost = init_cost
        self.normalize_baseline_rewards = normalize_baseline_rewards
        self.gradient_steps = gradient_steps
        # only for continuous policies parametrized by sigma
        self.update_cst_sigma = update_cst_sigma
        self.update_cst_sigma_kwargs = update_cst_sigma_kwargs
        self.alpha_entropy = alpha_entropy
        self.alpha_std  = alpha_std
        self.mixed_std = mixed_std
        self.exp_decay_std = exp_decay_std
        self.clip_rewards = clip_rewards
        self.reward_normalization = reward_normalization
    
    def init(self):
        single_obs, _ = self.val_env.reset()
        self.train_state = create_train_state(single_obs=single_obs, module=self.policy, apply_fn=self.vmap_sample,
                                rng=PRNGKey(self.seed), learning_rate=self.learning_rate, clip_norm=self.clip_norm,
                                lr_schedule_kwargs=self.lr_schedule_kwargs, every_k_schedule=self.every_k_schedule)
        self.init_baseline()

        self.step = 0
        self.rng_key = PRNGKey(self.seed)
        self.mean_cum_reward = None
        self.best_mean_cum_reward = None
        self.alpha_exp_decay_std = 1
        self.obs = None

    @property
    def hparams(self):
        # TODO check
        return {"validation_interval": self.validation_interval, "seed": self.seed,
                "baseline": self.baseline, "clip_norm": self.clip_norm, 
                "learning_rate": self.learning_rate, "policy_type": self.policy_type, "n_actions": self.n_actions}

    @hparams.setter
    # TODO check
    def hparams(self, value):
        self.validation_interval = value.get(
            "validation_interval", self.validation_interval)
        self.seed = value.get("seed", self.seed)
        self.baseline = value.get("baseline", self.baseline)
        self.clip_norm = value.get("clip_norm", self.clip_norm)
        self.learning_rate = value.get("learning_rate", self.learning_rate)
        self.policy_type = value.get("policy_type", self.policy_type)
        self.n_actions = value.get("n_actions", self.n_actions)

    def init_baseline(self):
        if self.baseline is None:
            return
        if self.baseline != "model":
            return
    
    def apply_actions(self, baseline_actions: Sequence):
        assert(len(baseline_actions) == self.n_actions)
        rewards = []
        for i in range(self.n_actions):
            baseline_action: H2MG = baseline_actions[i]
            _, baseline_reward, _, _, _obs = self.env.step(baseline_action)
            rewards.append(baseline_reward)
        return jnp.stack(rewards, axis=0)

    def compute_baseline(self, state, batch, rng, batch_size):
        baseline_actions = []
        with jax_device_context():
            baseline_actions, _, _ = self.vmap_sample(
                state.params, batch, split(rng, batch_size), n_action=self.n_actions)
        if self.n_actions == 1:
            baseline_actions = [baseline_actions]
        rewards = self.apply_actions(baseline_actions)
        if self.normalize_baseline_rewards:
            if self.alpha_std is not None:
                rewards = (rewards - jnp.mean(rewards, axis=0, keepdims=True)) / \
                    (self.alpha_std * jnp.std(rewards, axis=0, keepdims=True) + 1)
            elif self.mixed_std:
                rewards = (rewards - jnp.mean(rewards, axis=0, keepdims=True)) / \
                    (jnp.std(rewards, axis=0, keepdims=True)/2 + jnp.std(rewards, axis=0, keepdims=True).mean()/2 + 1e-8)
            elif self.exp_decay_std is not None:
                rewards = (rewards - jnp.mean(rewards, axis=0, keepdims=True)) / \
                    (1 + self.alpha_exp_decay_std * (jnp.std(rewards, axis=0, keepdims=True)-1) + 1e-8)
            elif self.reward_normalization == "normal":
                rewards = (rewards - jnp.mean(rewards, axis=0, keepdims=True)) / (jnp.std(rewards, axis=0, keepdims=True) + 1e-8)
            elif self.reward_normalization == "minmax":
                _min_rewards = jnp.min(rewards, axis=0, keepdims=True)
                _max_rewards = jnp.max(rewards, axis=0, keepdims=True)
                rewards = 2 * (rewards - _min_rewards) / (_max_rewards - _min_rewards + 1e-8) - 1
            else:
                raise ValueError("Wrong reward normalization configuration")

        if self.clip_rewards is not None:
            rewards = (rewards - jnp.mean(rewards, axis=0, keepdims=True))
            rewards = jnp.clip(rewards, a_min=-self.clip_rewards, a_max=self.clip_rewards)
        if self.baseline == "mean":
            baseline_rewards = jnp.mean(rewards, axis=0, keepdims=True)
        elif self.baseline == "median":
            baseline_rewards = jnp.median(rewards, axis=0, keepdims=True)
        else:
            baseline_rewards = 0
        return baseline_rewards, collate_h2mgs(baseline_actions), rewards

    def train_step(self, state: ReinforceTrainState, batch, *, rng, batch_size, step):
        baseline, baseline_actions, b_rewards = self.compute_baseline(
            state, batch, rng, batch_size)
        action, log_probs, policy_info = state.apply_fn(
            state.params, batch, split(rng, batch_size))
        next_obs, rewards, done, _, env_info = self.env.step(action)
        if step % self.max_steps == (self.max_steps-1):
            next_obs, info = self.env.reset(options={"load_new_power_grid": True})
        with jax_device_context():
            for _ in range(self.gradient_steps):
                (value, loss_info), grad = self.value_and_grad_fn(
                    state.params, batch, baseline_actions, b_rewards, baseline, multiple_actions=True)
                state = state.apply_gradients(grads=grad)
        algo_info = {"grad_norm": optax._src.linear_algebra.global_norm(grad),
                     "loss_value": value,
                     "log_probs": jnp.mean(log_probs),
                     "baseline": jnp.mean(baseline),
                     "mean_baseline_reward": jnp.mean(b_rewards),
                     "std_baseline_reward": jnp.std(b_rewards, axis=0).mean()} | loss_info
        if self.exp_decay_std is not None:
            algo_info =  algo_info | {"alpha_exp_decay_std": self.alpha_exp_decay_std,
                                      "exp_decay_std": self.exp_decay_std}
            self.alpha_exp_decay_std *= self.exp_decay_std
        return state, next_obs, rewards, policy_info, env_info, algo_info

    def validation_step(self, state, batch, *, rng, batch_size):
        action, _, sample_info = self.vmap_sample(
            self.policy_params, batch, split(rng, batch_size), determnistic=True)
        next_obs, rewards, done, _, step_info = self.val_env.step(action)
        return sample_info, step_info, rewards

    @property
    def policy_params(self):
        return self.train_state.params
    
    def _update_cst_sigma(self, step, alpha: float=1, period=1, end_sigma: float=None, **kwargs):
        if self.policy_type != "continuous":
            raise ValueError(f"Update sigma only for continuous policies, not {self.policy_type}")
        if self.update_cst_sigma == "exponential_decay":
            self.policy.cst_sigma = np.power(alpha, 1/period) * self.policy.cst_sigma
            if end_sigma is not None and end_sigma > self.policy.cst_sigma:
                self.policy.cst_sigma = max(self.policy.cst_sigma, end_sigma)
        else:
            raise NotImplementedError(f"{self.update_cst_sigma} not implemented.")

    def learn(self, logger=None, seed=None, batch_size=None, n_iterations=1000, validation_interval=None):
        validation_interval = validation_interval or self.validation_interval
        logger = logger or self.logger
        seed = seed or self.seed

        if self.rng_key is None:
            self.rng_key = PRNGKey(seed)
        if self.obs is None:
            self.obs = self.env.reset()[0]
        if self.best_mean_cum_reward is None or self.mean_cum_reward is None:
            self.mean_cum_reward, eval_infos = self.eval_reward()
            if self.best_mean_cum_reward is None:
                self.best_mean_cum_reward = self.mean_cum_reward
            logger.log_metrics_dict(
                        {"val_cumulative_reward": self.mean_cum_reward} | eval_infos, step=self.step)
            self.save_best_params(
                            self.run_dir, self.policy_params, step=-1, value=self.best_mean_cum_reward)
        for i in tqdm(range(self.step, n_iterations)):
            # Save training state
            self.step = i
            self.save(self.run_dir)
            
            # Train step
            self.rng_key, subkey = split(self.rng_key)
            self.train_state, self.obs, rewards, policy_info, env_info, algo_info = self.train_step(
                self.train_state, self.obs, rng=subkey, batch_size=batch_size, step=i)
            
            if self.update_cst_sigma is not None:
                self._update_cst_sigma(step=i, **self.update_cst_sigma_kwargs)

            # Logging
            logger.log_dicts(i, "train_", policy_info, env_info, algo_info)

            # Val step
            if i % validation_interval == (validation_interval - 1) and self.val_env is not None:
                self.val_env.reset()
                self.mean_cum_reward, eval_infos = self.eval_reward()
                # Save best params
                if self.mean_cum_reward >= self.best_mean_cum_reward:
                    self.best_mean_cum_reward = self.mean_cum_reward
                    self.save_best_params(
                        self.run_dir, self.policy_params, step=i, value=self.best_mean_cum_reward)
                logger.log_metrics_dict(
                    {"val_cumulative_reward": self.mean_cum_reward} | eval_infos, i)
        self.save_last_params(self.run_dir, self.policy_params,
                              step=i, value=self.mean_cum_reward)
        
        # Save last training state
        self.step = n_iterations
        self.save(self.run_dir)
        

    def test(self, test_env, res_dir, test_name=None, max_steps=None):
        test_name = "test_best"
        best_test_dir = os.path.join(res_dir, test_name)
        if not os.path.exists(best_test_dir):
            os.mkdir(best_test_dir)
        params = self.load_best_params()
        test_policy(test_env, self.policy, params,
                    seed=self.seed, output_dir=best_test_dir, max_steps=max_steps)
        test_name = "test_last"
        last_test_dir = os.path.join(res_dir, test_name)
        if not os.path.exists(last_test_dir):
            os.mkdir(last_test_dir)
        params = self.load_last_params()
        test_policy(test_env, self.policy, params,
                    seed=self.seed, output_dir=last_test_dir, max_steps=max_steps)

    def eval_reward(self):
        return eval_reward(self.val_env, self.policy, self.policy_params, seed=self.seed, max_steps=self.max_steps)

    @partial(jit, static_argnums=(0, 6))
    def value_and_grad_fn(self, params, observations, actions, rewards, baseline=0, multiple_actions=False):
        return value_and_grad(self.loss_fn, has_aux=True)(params, observations, actions, rewards, baseline, multiple_actions)

    def loss_fn(self, params, observations, actions, rewards, baseline=0, multiple_actions: bool = False):
        if multiple_actions:
            log_probs, log_prob_info = vmap(self.vmap_log_prob, in_axes=(
                None, None, 0), out_axes=0)(params, observations, actions)
        else:
            log_probs, log_prob_info = self.vmap_log_prob(
                params, observations, action=actions)
        reinforce_loss =  - (log_probs * (rewards - baseline)).mean()
        loss =  reinforce_loss
        # if self.entropy_loss is not None:
        entropy = self.policy.entropy(log_prob_info, batch=True, matrix=True, axis=1)
        if self.alpha_entropy is not None:
            entropy_loss = - self.alpha_entropy * entropy
            loss += entropy_loss
        else:
            entropy = 0
            entropy_loss = 0
        return loss, {"entropy": entropy, "entropy_loss": entropy_loss, "reinforce_loss": reinforce_loss}

    # @partial(jit, static_argnums=(0, 4, 5))
    def vmap_sample(self, params, obs, seed,
            deterministic=False, n_action=1) -> Tuple[Sequence[Any], Sequence[float], Dict[str, Any]]:
        return self.policy.vmap_sample(params, obs, seed, deterministic, n_action)

    # @partial(jit, static_argnums=(0,))
    def vmap_log_prob(self, params, obs, action):
        return self.policy.vmap_log_prob(params, obs, action)
    
    def eval(self, val_env, seed=None, max_steps=None):
        params = self.load_best_params()
        value, _ = eval_reward(val_env, self.policy, params, seed=seed, max_steps=max_steps)
        return value

    def _policy_filename(self, folder):
        return os.path.join(folder, "policy.pkl")

    def _hparams_filename(self, folder):
        return os.path.join(folder, "hparams.pkl")

    def _train_state_filename(self, folder):
        return os.path.join(folder, "train_state.pkl")

    def _params_filename(self, folder, step: int = None, value: float = None):
        filename = "params"
        if step is not None:
            filename += f"_{step}"
        if value is not None:
            filename += f"_{value:.4e}"
        return os.path.join(folder, f"{filename}.pkl")

    def load(self, folder):
        with open(os.path.join(folder, "reinforce.pkl"), "rb") as f:
            train_step = pickle.load(f)
            params = pickle.load(f)
            opt_state = pickle.load(f)
            self.step = pickle.load(f)
            self.rng_key = pickle.load(f)
            self.mean_cum_reward = pickle.load(f)
            self.best_mean_cum_reward = pickle.load(f)
            # self.obs = pickle.load(f)
            pickle.load(f) # load an obs
            self.alpha_exp_decay_std = pickle.load(f)
            self.policy.cst_sigma = pickle.load(f)
            states = pickle.load(f)
            np_randoms = pickle.load(f)
            self.val_env.state = pickle.load(f)
            self.val_env.np_random = pickle.load(f)
        self.train_state = self.train_state.replace(
            step=train_step,
            params=params,
            opt_state=opt_state)
        self.env.set_attr("state", states)
        self.env.set_attr("np_random", np_randoms)

    def save(self, folder):
        self.policy.save(self._policy_filename(folder))
        with open(self._params_filename(folder), 'wb') as f:
            pickle.dump(self.policy_params, f)
        with open(os.path.join(folder, "reinforce.pkl"), "wb") as f:
            pickle.dump(self.train_state.step, f)
            pickle.dump(self.train_state.params, f)
            pickle.dump(self.train_state.opt_state, f)
            pickle.dump(self.step, f)
            pickle.dump(self.rng_key, f)
            pickle.dump(self.mean_cum_reward, f)
            pickle.dump(self.best_mean_cum_reward, f)
            pickle.dump(self.obs, f)
            pickle.dump(self.alpha_exp_decay_std, f)
            pickle.dump(self.policy.cst_sigma, f)
            pickle.dump(self.env.get_attr("state"), f)
            pickle.dump(self.env.get_attr("np_random"), f)
            pickle.dump(self.val_env.state, f)
            pickle.dump(self.val_env.np_random, f)

    def save_best_params(self, folder, params=None, step=None, value=None):
        if params is None:
            params = self.policy_params
        with open(self.best_params_path, 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(folder, self.best_params_info_name), 'w') as f:
            json.dump({"step": step, "value": value}, f)

    def save_last_params(self, folder, params=None, step=None, value=None):
        if params is None:
            params = self.policy_params
        with open(self.last_params_path, 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(folder, 'last_params_info.json'), 'w') as f:
            json.dump({"step": step, "value": value}, f)

    def load_best_params(self) -> dict:
        with open(self.best_params_path, 'rb') as f:
            params = pickle.load(f)
        return params

    def load_last_params(self) -> dict:
        with open(self.last_params_path, 'rb') as f:
            params = pickle.load(f)
        return params
