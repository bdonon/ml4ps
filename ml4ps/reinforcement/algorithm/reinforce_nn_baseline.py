import pickle
from functools import partial
from typing import Any, Dict, List, Sequence, Tuple

import jax.numpy as jnp
import optax
from flax.training import train_state
from gymnasium.vector.utils.spaces import iterate
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.random import PRNGKey, split
from jax.tree_util import tree_leaves
from ml4ps.h2mg import H2MG, H2MGStructure, HyperEdgesStructure
from ml4ps.logger import BaseLogger
from ml4ps.neural_network import H2MGNODE
from ml4ps.neural_network import get as get_neural_network
from ml4ps.reinforcement.environment import PSBaseEnv
from ml4ps.reinforcement.policy import BasePolicy

from .reinforce import Reinforce


class ReinforceTrainState(train_state.TrainState):
    pass


def create_train_state(*, env, module, apply_fn, rng, learning_rate, clip_norm=0.1) -> ReinforceTrainState:
    """Creates an initial `TrainState`."""
    batch_obs, _ = env.reset()
    single_obs = next(iterate(env.observation_space, batch_obs))
    params = module.init(rng, single_obs)
    tx = optax.chain(optax.clip_by_global_norm(clip_norm),
                     optax.adam(learning_rate=learning_rate))
    return ReinforceTrainState.create(apply_fn=apply_fn, params=params, tx=tx)


def count_params(params):
    return sum(x.size for x in tree_leaves(params))

# TODO: merge nn_baseline with Reinforce: function can have (state, baseline_state=None) as input


class ReinforceBaseline(Reinforce):
    policy: BasePolicy
    baseline_model: H2MGNODE
    env: PSBaseEnv
    logger: BaseLogger
    train_state: ReinforceTrainState

    def __init__(self, env: PSBaseEnv, seed=0, *, val_env: PSBaseEnv, test_env: PSBaseEnv, run_dir, max_steps, policy_type: str,
                 logger=None, validation_interval=100, baseline, policy_args={}, nn_args={}, clip_norm, learning_rate,
                 baseline_nn_type="h2mgnode", n_actions=None, baseline_nn_args=None, baseline_learning_rate, init_cost, nn_baseline_steps) -> 'ReinforceBaseline':
        super().__init__(env, seed, val_env=val_env, test_env=test_env, run_dir=run_dir, max_steps=max_steps, policy_type=policy_type,
                         logger=logger, validation_interval=validation_interval, baseline=baseline, policy_args=policy_args, nn_args=nn_args,
                         clip_norm=clip_norm, learning_rate=learning_rate, baseline_learning_rate=baseline_learning_rate, n_actions=n_actions, baseline_nn_type=baseline_nn_type, baseline_nn_args=baseline_nn_args, init_cost=init_cost, nn_baseline_steps=nn_baseline_steps)

    @property
    def hparams(self):
        return {"validation_interval": self.validation_interval, "seed": self.seed, "baseline": self.baseline,
                "baseline_nn_type": self.baseline_nn_type, "baseline_nn_args": self.baseline_nn_args, "baseline_learning_rate": self.baseline_learning_rate}

    @property
    def policy_params(self):
        return self.train_state["policy"].params

    def init(self):
        self.init_baseline()
        policy_state = create_train_state(env=self.env, module=self.policy, apply_fn=self.vmap_sample, rng=PRNGKey(
            self.seed), learning_rate=self.learning_rate, clip_norm=self.clip_norm)
        baseline_state = create_train_state(env=self.env, module=self.baseline_model, apply_fn=self.apply_baseline, rng=PRNGKey(
            self.seed), learning_rate=self.baseline_learning_rate, clip_norm=self.clip_norm)
        self.train_state = {"policy": policy_state, "baseline": baseline_state}

    def init_baseline(self):
        if self.baseline_nn_args is None:
            self.baseline_nn_args = {}
        output_structure = H2MGStructure()
        output_structure.add_global_hyper_edges_structure(
            HyperEdgesStructure(features={"baseline": 1}))
        self.baseline_model = get_neural_network(
            self.baseline_nn_type, **{"output_structure": output_structure, **self.baseline_nn_args})

    @partial(jit, static_argnums=(0,))
    def apply_baseline(self, params: dict, batch):
        # .global_hyper_edges.features["baseline"]
        return vmap(self._apply_baseline, in_axes=(None, 0), out_axes=0)(params, batch)

    def _apply_baseline(self, params, batch):
        batch = self.policy.normalizer(batch)
        # TODO: Change
        return 0.1 * self.baseline_model.apply(params, batch).global_hyper_edges.features["baseline"]

    @partial(jit, static_argnums=(0,))
    def apply_h1(self, params: dict, batch):
        return vmap(self._apply_h1, in_axes=(None, 0), out_axes=0)(params, batch)

    def _apply_h1(self, params, batch):
        batch = self.policy.normalizer(batch)
        return self.baseline_model.apply_h1(params, batch)

    def compute_baseline(self, state, batch, rng, batch_size, baseline_state=None):
        _, baseline_actions, b_rewards = super().compute_baseline(state,
                                                                  batch, rng, batch_size)
        baseline = jnp.squeeze(self.apply_baseline(
            baseline_state.params, batch), axis=1)
        return baseline, baseline_actions, b_rewards

    def train_step(self, state: ReinforceTrainState, batch: H2MG, *, rng, batch_size, step):
        policy_state, baseline_state = state["policy"], state["baseline"]
        baseline, baseline_actions, b_rewards = self.compute_baseline(
            policy_state, batch, rng, batch_size, baseline_state)
        action, log_probs, sample_info = policy_state.apply_fn(
            policy_state.params, batch, split(rng, batch_size))
        next_obs, rewards, done, _, step_info = self.env.step(action)
        if step % self.max_steps == (self.max_steps-1):
            next_obs, info = self.env.reset(
                options={"load_new_power_grid": True})
        policy_loss_value, policy_grad = self.value_and_grad_fn(
            policy_state.params, batch, baseline_actions, b_rewards, baseline, multiple_actions=True)
        policy_state = policy_state.apply_gradients(grads=policy_grad)
        for _ in range(self.nn_baseline_steps):
            baseline_loss_value, baseline_grad = self.baseline_value_and_grad_fn(
                baseline_state.params, batch, b_rewards.mean(axis=0))
            baseline_state = baseline_state.apply_gradients(
                grads=baseline_grad)
        assert(baseline.shape == jnp.mean(b_rewards, axis=0).shape)
        other_info = {"grad_norm": optax._src.linear_algebra.global_norm(policy_grad),
                      "loss_value": policy_loss_value,
                      "log_probs": jnp.mean(log_probs),
                      "baseline": jnp.mean(baseline),
                      "std_baseline": jnp.std(baseline),
                      "mean_reward": jnp.mean(b_rewards),
                      "std_baseline_reward": jnp.std(b_rewards, axis=0).mean(),
                      "baseline_grad_norm": optax._src.linear_algebra.global_norm(baseline_grad),
                      "baseline_loss_value": baseline_loss_value,
                      "baseline_abs_diff": jnp.abs(baseline - jnp.mean(b_rewards, axis=0)),
                      "baseline_l2_diff": (baseline - jnp.mean(b_rewards, axis=0))**2}
        return {"policy": policy_state, "baseline": baseline_state}, next_obs, rewards, sample_info, step_info, other_info

    @partial(jit, static_argnums=(0,))
    def baseline_value_and_grad_fn(self, params, observations, rewards):
        return value_and_grad(self.baseline_loss_fn)(params, observations, rewards)

    def baseline_loss_fn(self, params, observations, rewards):
        baseline = jnp.squeeze(self.apply_baseline(
            params, observations), axis=1)
        return jnp.square(rewards - baseline).mean()

    def save(self, folder):
        self.policy.save(self._policy_filename(folder))
        with open(self._params_filename(folder), 'wb') as f:
            pickle.dump(self.train_state["policy"].params, f)
