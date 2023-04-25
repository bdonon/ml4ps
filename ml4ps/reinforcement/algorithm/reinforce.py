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

from .algorithm import Algorithm


class ReinforceTrainState(train_state.TrainState):
    pass


def create_train_state(*, single_obs, module, apply_fn, rng, learning_rate, clip_norm=0.1) -> ReinforceTrainState:
    """Creates an initial `TrainState`."""
    params = module.init(rng, single_obs)
    # TODO: clip_by_values ?
    tx = optax.chain(optax.clip_by_global_norm(clip_norm),
                     optax.adam(learning_rate=learning_rate))
    # tx = optax.MultiSteps(tx, every_k_schedule=2) # TODO: REMOVE
    return ReinforceTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def count_params(params):
    return sum(x.size for x in tree_leaves(params))


class Reinforce(Algorithm):
    policy: BasePolicy
    env: PSBaseEnv
    logger: BaseLogger
    train_state: ReinforceTrainState

    def __init__(self, env: PSBaseEnv, seed=0, *, val_env: PSBaseEnv, test_env: PSBaseEnv, run_dir, max_steps, policy_type: str, logger=None, validation_interval=100, baseline=None, policy_args={}, nn_args={}, clip_norm, learning_rate, n_actions=5, baseline_learning_rate=None, baseline_nn_type=None, baseline_nn_args=None) -> 'Reinforce':
        super().__init__()
        self.best_params_path = os.path.join(run_dir, "best_params.pkl")
        self.last_params_path = os.path.join(run_dir, "last_params.pkl")
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
        self.baseline_learning_rate = baseline_learning_rate
        self.baseline_nn_type = baseline_nn_type
        self.baseline_nn_args = baseline_nn_args
        self.validation_interval = validation_interval
        self.n_actions = n_actions
        self.run_dir = run_dir
        self.init()

    def init(self):
        single_obs, _ = self.val_env.reset()
        self.train_state = create_train_state(single_obs=single_obs, module=self.policy, apply_fn=self.vmap_sample, rng=PRNGKey(
            self.seed), learning_rate=self.learning_rate, clip_norm=self.clip_norm)
        self.init_baseline()

    @property
    def hparams(self):
        # TODO check
        return {"validation_interval": self.validation_interval, "seed": self.seed,
                "baseline": self.baseline, "clip_norm": self.clip_norm, "learning_rate": self.learning_rate, "policy_type": self.policy_type, "n_actions": self.n_actions}

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

    def compute_baseline(self, state, batch, rng, batch_size):
        rewards = []
        baseline_actions = []
        baseline_actions, _, _ = self.vmap_sample(
            state.params, batch, split(rng, batch_size), n_action=self.n_actions)
        for i in range(self.n_actions):
            baseline_action = baseline_actions[i]
            _, baseline_reward, _, _, _ = self.env.step(baseline_action)
            rewards.append(baseline_reward)
        rewards = jnp.stack(rewards, axis=0)
        if self.baseline == "mean":
            baseline_rewards = jnp.mean(rewards, axis=0)
        elif self.baseline == "median":
            baseline_rewards = jnp.median(rewards, axis=0)
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
            next_obs, info = self.env.reset(
                options={"load_new_power_grid": True})
        value, grad = self.value_and_grad_fn(
            state.params, batch, baseline_actions, b_rewards, baseline, multiple_actions=True)
        state = state.apply_gradients(grads=grad)
        algo_info = {"grad_norm": optax._src.linear_algebra.global_norm(grad),
                     "loss_value": value,
                     "log_probs": jnp.mean(log_probs),
                     "baseline": jnp.mean(baseline),
                     "mean_baseline_reward": jnp.mean(b_rewards),
                     "std_baseline_reward": jnp.std(b_rewards, axis=0).mean()}
        return state, next_obs, rewards, policy_info, env_info, algo_info

    def validation_step(self, state, batch, *, rng, batch_size):
        action, _, sample_info = self.vmap_sample(
            self.policy_params, batch, split(rng, batch_size), determnistic=True)
        next_obs, rewards, done, _, step_info = self.val_env.step(action)
        return sample_info, step_info, rewards

    @property
    def policy_params(self):
        return self.train_state.params

    def learn(self, logger=None, seed=None, batch_size=None, n_iterations=1000, validation_interval=None):
        best_mean_cum_reward = -np.inf
        validation_interval = validation_interval or self.validation_interval
        logger = logger or self.logger
        seed = seed or self.seed

        rng_key = PRNGKey(seed)
        rng_key, rng_key_val = split(rng_key)
        obs, _ = self.env.reset()
        self.val_env.reset()
        for i in tqdm(range(n_iterations)):
            # Train step
            rng_key, subkey = split(rng_key)
            self.train_state, obs, rewards, policy_info, env_info, algo_info = self.train_step(
                self.train_state, obs, rng=subkey, batch_size=batch_size, step=i)

            # Logging
            logger.log_dicts(i, "train_", policy_info, env_info, algo_info)

            # Val step
            if i % validation_interval == 0 and self.val_env is not None:
                self.val_env.reset()
                mean_cum_reward = self.eval_reward()
                last_value = mean_cum_reward
                last_step = i
                # Save best params
                if mean_cum_reward >= best_mean_cum_reward:
                    best_mean_cum_reward = mean_cum_reward
                    self.save_best_params(
                        self.run_dir, self.policy_params, step=i, value=best_mean_cum_reward)
                logger.log_metrics_dict(
                    {"val_cumulative_reward": mean_cum_reward}, i)
        self.save_last_params(self.run_dir, self.policy_params,
                              step=last_step, value=last_value)

    def test(self, test_env, res_dir, test_name=None):
        test_name = "test_best"
        best_test_dir = os.path.join(res_dir, test_name)
        if not os.path.exists(best_test_dir):
            os.mkdir(best_test_dir)
        params = self.load_best_params()
        test_policy(test_env, self.policy, params,
                    seed=self.seed, output_dir=best_test_dir)
        test_name = "test_last"
        last_test_dir = os.path.join(res_dir, test_name)
        if not os.path.exists(last_test_dir):
            os.mkdir(last_test_dir)
        params = self.load_last_params()
        test_policy(test_env, self.policy, params,
                    seed=self.seed, output_dir=last_test_dir)

    def eval_reward(self):
        return eval_reward(self.val_env, self.policy, self.policy_params, seed=self.seed, n=100)

    @partial(jit, static_argnums=(0, 6))
    def value_and_grad_fn(self, params, observations, actions, rewards, baseline=0, multiple_actions=False):
        return value_and_grad(self.loss_fn)(params, observations, actions, rewards, baseline, multiple_actions)

    def loss_fn(self, params, observations, actions, rewards, baseline=0, multiple_actions: bool = False):
        if multiple_actions:
            log_probs = vmap(self.vmap_log_prob, in_axes=(
                None, None, 0), out_axes=0)(params, observations, actions)
        else:
            log_probs = self.vmap_log_prob(
                params, observations, action=actions)
        return - (log_probs * (rewards - baseline)).mean()

    @partial(jit, static_argnums=(0, 4, 5))
    def vmap_sample(self, params, obs, seed, deterministic=False, n_action=1) -> Tuple[Sequence[Any], Sequence[float], Dict[str, Any]]:
        return self.policy.vmap_sample(params, obs, seed, deterministic, n_action)

    @partial(jit, static_argnums=(0,))
    def vmap_log_prob(self, params, obs, action):
        return vmap(self.policy.log_prob, in_axes=(None, 0, 0), out_axes=0)(params, obs, action)

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
        with open(self._hparams_filename(folder), 'rb') as f:
            self.hparams = pickle.load(f)
        self.policy = get_policy(
            self.policy_type, None, file=self._policy_filename(folder))

    def save(self, folder):
        self.policy.save(self._policy_filename(folder))
        with open(self._params_filename(folder), 'wb') as f:
            pickle.dump(self.policy_params, f)

    def save_best_params(self, folder, params=None, step=None, value=None):
        if params is None:
            params = self.policy_params
        with open(self.best_params_path, 'wb') as f:
            pickle.dump(params, f)
        with open(os.path.join(folder, 'best_params_info.json'), 'w') as f:
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
