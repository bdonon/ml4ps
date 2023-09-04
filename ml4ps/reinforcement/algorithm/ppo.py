import json
import os
import pickle
from collections import defaultdict, namedtuple
from dataclasses import dataclass
from functools import partial
from time import time
from typing import (Any, Callable, Dict, Iterator, List, NamedTuple, Optional,
                    Sequence, Tuple, Union)

import gymnasium as gym
import jax
import jax.numpy as jnp
import numpy as np
import optax
from gymnasium.vector.utils.spaces import iterate
from jax import jit
from jax import numpy as jnp
from jax import value_and_grad, vmap
from jax.random import KeyArray, PRNGKey, PRNGKeyArray, split
from ml4ps.h2mg import (H2MG, H2MGStructure, HyperEdgesStructure,
                        collate_h2mgs, separate_h2mgs)
from ml4ps.logger import (BaseLogger, dict_mean, mean_of_dicts,
                          process_venv_dict)
from ml4ps.neural_network import H2MGNODE
from ml4ps.neural_network import get as get_neural_network
from ml4ps.reinforcement.environment import PSBaseEnv
from ml4ps.reinforcement.policy import BasePolicy, get_policy
from ml4ps.reinforcement.test_policy import eval_reward, test_policy
from tqdm import tqdm

from .algorithm import Algorithm


@dataclass
class PPOTuple:
    state: H2MG
    action: H2MG
    reward: float
    next_state: H2MG
    done: bool
    prev_log_prob: Optional[float] = None
    advantage: Optional[float] = None
    next_value: Optional[float] = None
    value: Optional[float] = None


class BatchedPPOTuples(NamedTuple):
    batch_size: int
    state: H2MG
    action: H2MG
    reward: Sequence[float]
    next_state: H2MG
    done: Sequence[bool]
    truncated: Sequence[bool]
    terminated: Sequence[bool]
    info: Any
    prev_log_prob: Optional[Sequence[float]] = None
    advantage: Optional[Sequence[float]] = None
    next_value: Optional[Sequence[float]] = None
    value: Optional[Sequence[float]] = None
    returns: Optional[Sequence[float]] = None


def separate_info(info_dict: Dict[str, np.ndarray]) -> List[Dict[str, float]]:
    lengths = {}
    lengths_set = set()
    for k, value in info_dict.items():
        if isinstance(value, Sequence):
            value_len = len(value)
        else:
            value_len = 1
        lengths[k] = value_len
        lengths_set.add(value_len)
    assert(len(lengths_set) <= 2)
    info_dicts = [info_dict for _ in range(max(lengths_set))]
    for k, value in info_dict.items():
        for i, info_d in enumerate(info_dicts):
            if isinstance(value, np.array):
                info_d[k] = value[i]
            else:
                info_d[k] = value
    return info_dicts


def batched_tuple_to_sequence_of_tuples(batched_tuple: BatchedPPOTuples) -> Sequence[BatchedPPOTuples]:
    # TODO test it
    batch_sizes = np.ones(shape=(batched_tuple.batch_size,))
    states = separate_h2mgs(batched_tuple.state)
    actions = separate_h2mgs(batched_tuple.action)
    rewards = batched_tuple.reward
    next_states = separate_h2mgs(batched_tuple.next_state)
    dones = batched_tuple.done
    truncateds = batched_tuple.truncated
    terminateds = batched_tuple.terminated
    infos = separate_info(batched_tuple.info)
    prev_log_probs = batched_tuple.prev_log_prob
    advantages = batched_tuple.advantage
    next_values = batched_tuple.next_value
    values = batched_tuple.value
    returns = batched_tuple.returns

    for batch_size, state, action, reward, next_state, done, truncated, terminated, \
        info, prev_log_prob, advantage, next_value, value, return_ in \
            zip(batch_sizes, states, actions, rewards, next_states, dones, truncateds,
                terminateds, infos, prev_log_probs, advantages, next_values, values, returns):
        BatchedPPOTuples(batch_size=batch_size,
                         state=state,
                         action=action,
                         reward=reward,
                         next_state=next_state,
                         terminated=terminated,
                         truncated=truncated,
                         done=done,
                         info=info,
                         prev_log_prob=prev_log_prob,
                         advantage=advantage,
                         value=value,
                         next_value=next_value,
                         returns=return_)


def sequence_of_tuples_to_batched_tuple(sequence_of_tuples: Sequence[BatchedPPOTuples]) -> BatchedPPOTuples:
    # TODO
    raise NotImplementedError
    batch_sizes = np.ones(shape=(batched_tuple.batch_size,))
    states = separate_h2mgs(batched_tuple.state)
    actions = separate_h2mgs(batched_tuple.action)
    rewards = batched_tuple.reward
    next_states = separate_h2mgs(batched_tuple.next_state)
    dones = batched_tuple.done
    truncateds = batched_tuple.truncated
    terminateds = batched_tuple.terminated
    infos = separate_info(batched_tuple.info)
    prev_log_probs = batched_tuple.prev_log_prob
    advantages = batched_tuple.advantage
    next_values = batched_tuple.next_value
    values = batched_tuple.value
    returns = batched_tuple.returns

    for batch_size, state, action, reward, next_state, done, truncated, terminated, \
        info, prev_log_prob, advantage, next_value, value, return_ in \
            zip(batch_sizes, states, actions, rewards, next_states, dones, truncateds,
                terminateds, infos, prev_log_probs, advantages, next_values, values, returns):
        BatchedPPOTuples(batch_size=batch_size,
                         state=state,
                         action=action,
                         reward=reward,
                         next_state=next_state,
                         terminated=terminated,
                         truncated=truncated,
                         done=done,
                         info=info,
                         prev_log_prob=prev_log_prob,
                         advantage=advantage,
                         value=value,
                         next_value=next_value,
                         returns=return_)


class BatchedPPOEpisodes:
    def __init__(self, batch_size, max_episode_length=None):
        if batch_size < 1:
            raise ValueError("batch_size should be at least 1")
        self.batch_size = batch_size
        self.max_episode_length = max_episode_length
        self.batched_transitions = []

    def add(self, batched_transition: BatchedPPOTuples) -> None:
        if batched_transition.batch_size != self.batch_size:
            raise ValueError(
                f"Expected {self.batch_size} transition, got {batched_transition.batch_size}")
        if self.max_episode_length is not None and len(self.batched_transitions) < self.max_episode_length:
            self.batched_transitions.append(batched_transition)

    def len(self) -> int:
        return len(self.batched_transitions)

    def __iter__(self) -> Iterator:
        return iter(self.batched_transitions)

    def set_transitions(self, batched_transitions: List[BatchedPPOTuples]):
        self.batched_transitions = batched_transitions


class EpisodesCollection:
    episodes_list: List[BatchedPPOEpisodes]

    def __init__(self, episodes_list: List[BatchedPPOEpisodes] = None):
        self.episodes_list = episodes_list if episodes_list is not None else []

    def append(self, episode: BatchedPPOEpisodes):
        self.episodes_list.append(episode)

    def __iter__(self):
        for episodes in self.episodes_list:
            for batch in episodes:
                yield batch

    def map(self, fn: Callable):
        self.episodes_list = list(map(fn, self.episodes_list))

    def shuffled(self, rng):
        indexes = jnp.arange(len(self.episodes_list))
        shuffled_indexes = jax.random.permutation(
            rng, indexes, independent=True)
        for idx in shuffled_indexes:
            rng, sub_rng = split(rng)
            batch_list = list(self.episodes_list[idx])
            batch_indexes = jax.random.permutation(
                sub_rng, jnp.arange(len(batch_list)), independent=True)
            for batch_idx in batch_indexes:
                yield batch_list[batch_idx]


class ValueNetwork:
    net: H2MGNODE

    def __init__(self, venv, normalizer, nn_args, scaling_factor=1.0):
        output_structure = H2MGStructure()
        output_structure.add_global_hyper_edges_structure(
            HyperEdgesStructure(features={"value": 1}))
        self.net = get_neural_network(
            "h2mgnode", **{"output_structure": output_structure}, **nn_args)
        self.normalizer = normalizer
        self.scaling_factor = scaling_factor

    def init(self, rng, obs):
        return self.net.init(rng, obs)

    def __call__(self, params, obs) -> Any:
        return self.net.apply(params, obs)
    
    def apply(self, params, obs) -> Any:
        obs = self.normalizer(obs)
        return self.scaling_factor * self.net.apply(params, obs).global_hyper_edges.features["value"].squeeze()

    def vmap_apply(self, params, obs) -> Any:
        obs = self.normalizer(obs)
        return self.scaling_factor * \
            vmap(self.net.apply, in_axes=(None, 0), out_axes=0)(
                params, obs).global_hyper_edges.features["value"].squeeze()


class GAE:
    def __init__(self, value_network, gamma=0.99, lmbda=1.0, average_gae=True):
        self.value_network: ValueNetwork = value_network
        self.gamma = gamma
        self.lmbda = lmbda
        self.average_gae = average_gae

    @partial(jit, static_argnums=(0,))
    def compute_tr_adv(self, params: Dict, transition: BatchedPPOTuples, next_advantage: float):
        next_state = transition.next_state
        state = transition.state
        next_value = self.value_network.vmap_apply(params, next_state)
        value = self.value_network.vmap_apply(params, state)
        delta = transition.reward + self.gamma * \
            (next_value * (1-transition.done)) - \
            value  # TODO: check parenthsise
        advantage = delta + (self.gamma * self.lmbda *
                             next_advantage * (1-transition.done))
        return BatchedPPOTuples(batch_size=transition.batch_size,
                                state=transition.state,
                                action=transition.action,
                                reward=transition.reward,
                                next_state=transition.next_state,
                                terminated=transition.terminated,
                                truncated=transition.truncated,
                                done=transition.done,
                                info=transition.info,
                                prev_log_prob=transition.prev_log_prob,
                                advantage=advantage,
                                value=value,
                                next_value=next_value,
                                returns=advantage+value)

    def compute_gae(self, params, episodes: BatchedPPOEpisodes, lazy=True) -> None:
        next_advantage = 0
        r_transitions = []
        for transition in reversed(list(episodes)):
            transition = self.compute_tr_adv(
                params, transition, next_advantage)
            r_transitions.append(transition)
            next_advantage = transition.advantage

        new_episodes = BatchedPPOEpisodes(
            batch_size=episodes.batch_size, max_episode_length=episodes.max_episode_length)
        new_episodes.batched_transitions = list(reversed(r_transitions))
        return new_episodes

    # @partial(jit, static_argnums=(0,)) # TODO test if possible
    def compute_batch_gae(self, params, episodes: BatchedPPOEpisodes) -> None:
        return self.compute_gae(params, episodes)


def single_obs_from_venv(venv):
    batch_obs, _ = venv.reset()
    single_obs = next(iterate(venv.observation_space, batch_obs))
    return single_obs


def convert(d: Dict):
    return dict(map(lambda t: (t[0], jnp.array(t[1], dtype=jnp.float32)), d.items()))


def sample_batched_episode(*, policy: BasePolicy, params: Dict, rng: KeyArray, deterministic: bool = False,
                           env: gym.vector.VectorEnv, max_episode_length: int) -> BatchedPPOEpisodes:
    batch_size = env.num_envs
    batched_episodes = BatchedPPOEpisodes(
        batch_size=batch_size, max_episode_length=max_episode_length)
    obs, _ = env.reset(options={"load_new_power_grid": True})

    states = env.get_attr("state")
    names = [state.power_grid.name for state in states]

    cum_reward = 0
    stats = defaultdict(float)
    for i in range(max_episode_length):
        rng, sub_rng = split(rng)
        action, log_prob, action_info = policy.vmap_sample(
            params, obs, rng=split(sub_rng, batch_size), deterministic=deterministic)
        new_obs, reward, terminated, truncated, info = env.step(action)
        cum_reward += reward * (1 - terminated)
        # if i == max_episode_length-1: # TODO: Finite horizon or not ?
        #     truncated = True
        # TODO = info = convert(action_info | info) vs process_vevn_dict
        batched_transition = BatchedPPOTuples(batch_size=batch_size, state=obs, action=action, reward=reward,
                                              next_state=new_obs, done=jnp.logical_or(terminated, truncated),
                                              terminated=terminated, truncated=truncated, info=convert(action_info),
                                              prev_log_prob=log_prob, advantage=None, value=None, next_value=None, returns=None)
        batched_episodes.add(batched_transition)
        obs = new_obs

    for k, v in zip(names, cum_reward):
        stats[k] = v

    state = batched_transition.state
    gen_vm_pu = state.local_hyper_edges["gen"].features["vm_pu"]
    next_gen_vm_pu = batched_transition.next_state.local_hyper_edges["gen"].features["vm_pu"]
    return batched_episodes, {**batched_transition.info, "cum_reward": cum_reward.mean(), "gen_vm_pu_mean": gen_vm_pu.mean(),
                              "gen_vm_pu_min": gen_vm_pu.min(), "gen_vm_pu_max": gen_vm_pu.max(),
                              "next_gen_vm_pu_mean": next_gen_vm_pu.mean(), "next_gen_vm_pu_min": next_gen_vm_pu.min(),
                              "next_gen_vm_pu_max": next_gen_vm_pu.max()}  # | stats


def sample_episodes_collection(*, n: int, policy: BasePolicy, params: Dict, rng: KeyArray, deterministic: bool = False,
                               env: gym.vector.VectorEnv, max_episode_length: int) -> Tuple[EpisodesCollection, Dict]:
    episode_collection = EpisodesCollection()
    infos = []
    for _ in range(n):
        episodes, info = sample_batched_episode(
            policy=policy, params=params, rng=rng, deterministic=deterministic, env=env, max_episode_length=max_episode_length)
        infos.append(dict_mean(info))
        # env.reset(options={"load_new_power_grid": True})
        episode_collection.append(episodes)
    return episode_collection, mean_of_dicts(infos)


class PPO(Algorithm):
    env: gym.vector.VectorEnv
    value_network: ValueNetwork
    policy_network: BasePolicy
    policy_params: Dict
    value_params: Dict
    gae: GAE
    policy_optmizer: optax.GradientTransformation
    value_optimizer: optax.GradientTransformation
    logger: BaseLogger

    def __init__(self, env: gym.vector.VectorEnv, seed=None, policy_type="continuous", value_learning_rate=3e-4,
                 policy_learning_rate=3e-4, policy_clip_norm=1.0, value_clip_norm=0.1, every_k_schedule=None, eps=0.2,
                 gamma=0.99, lmbda=0.95, value_scaling_factor=0.1, cst_sigma=1e-3, policy_nn_args=None, value_nn_args=None,
                 logger=None, val_env=None, test_env=None, run_dir=None,  alpha_entropy=0):
        rng = PRNGKey(seed)
        self.seed = seed
        value_rng, policy_rng = split(rng)
        self.eps = eps
        self.env = env
        self.val_env = val_env
        self.test_env = test_env  # TODO: implement test on test_env
        self.run_dir = run_dir
        self.policy_type = policy_type
        self.policy_network, self.policy_params = self.init_policy_network(
            env, policy_rng, policy_type=policy_type, cst_sigma=cst_sigma, nn_args=policy_nn_args)
        self.policy_optmizer = optax.chain(optax.clip_by_global_norm(policy_clip_norm),
                                           optax.adam(learning_rate=policy_learning_rate))
        if every_k_schedule is not None and every_k_schedule > 1:
            self.policy_optmizer = optax.MultiSteps(
                self.policy_optmizer, every_k_schedule=every_k_schedule)  # Accumulate gradient for k batches
        self.policy_state = self.policy_optmizer.init(self.policy_params)
        self.value_network, self.value_params = self.init_value_network(
            env, value_rng, self.policy_network.normalizer, nn_args=value_nn_args, scaling_factor=value_scaling_factor)
        self.value_optimizer = optax.chain(optax.clip_by_global_norm(value_clip_norm),
                                           optax.adam(learning_rate=value_learning_rate))
        if every_k_schedule is not None and every_k_schedule > 1:
            self.value_optimizer = optax.MultiSteps(
                self.value_optimizer, every_k_schedule=every_k_schedule)  # Accumulate gradient for k batches
        self.value_state = self.value_optimizer.init(self.value_params)
        self.gae = GAE(value_network=self.value_network,
                       gamma=gamma, lmbda=lmbda)
        self.gamma = gamma
        self.lmbda = lmbda
        self.alpha_entropy = alpha_entropy
        self.logger = logger

    def init_value_network(self, env, rng, normalizer, nn_args, scaling_factor) -> Tuple[ValueNetwork, Dict]:
        single_obs = single_obs_from_venv(env)
        value_net = ValueNetwork(
            env, normalizer, nn_args=nn_args, scaling_factor=scaling_factor)
        params = value_net.init(rng, single_obs)
        return value_net, params

    def init_policy_network(self, env, rng, policy_type, cst_sigma, nn_args) -> Tuple[BasePolicy, Dict]:
        single_obs = single_obs_from_venv(env)
        policy = get_policy(
            policy_type, env, cst_sigma=cst_sigma, nn_args=nn_args)
        params = policy.init(rng, single_obs)
        return policy, params

    def learn(self, *, N: int, M: int, T: int, K: int, seed: int, batch_size: int = None, n_iterations: int,
              env: gym.vector.VectorEnv = None, logger=None, validation_interval=100, log_interval=10, deterministic_exploration=False, learn_policy=True):
        rng = PRNGKey(seed)
        self.logger = logger or self.logger
        env = env or self.env
        best_mean_cum_reward = -np.inf
        if M != env.num_envs:
            raise ValueError(
                F"The mini batch size M ({M}) should be equal to the number of parallel environments ({env.num_envs})")
        if batch_size is not None and batch_size != M:
            raise ValueError(
                F"The mini batch size M ({M}) should be equal to the minibatch size ({batch_size})")
        if N % env.num_envs != 0:
            raise ValueError(
                f"N={N}, the number of trajectories should be a multiple of the \
                     number of parallel environments env.num_envs={env.num_envs}")
        # number of batched trajectories = number of trajectories / number of parallel environments
        n = N // env.num_envs
        max_episode_length = T
        epochs = K

        step = 0
        mean_cum_reward, best_mean_cum_reward, eval_info = self.eval_step(
            self.val_env, self.policy_network, self.policy_params, rng, max_episode_length, step, best_mean_cum_reward)
        self.log({"val_cumulative_reward": mean_cum_reward} | eval_info, step)
        for i in tqdm(range(n_iterations)):

            # Sample data (n batched episodes of length T or N episodes of length T)
            episodes_collection, infos = sample_episodes_collection(
                n=n, policy=self.policy_network, params=self.policy_params, rng=rng, deterministic=deterministic_exploration, # TODO: change back  deterministic=False
                env=env, max_episode_length=max_episode_length)
            if step % log_interval == 0:
                self.log(infos, step=step)

            # Compute advantages
            episodes_collection.map(lambda batched_episodes: self.gae.compute_batch_gae(
                self.value_params, batched_episodes))

            # Learn policy
            for _ in range(epochs):
                rng, sub_rng = split(rng)
                for batch in episodes_collection.shuffled(rng=sub_rng):
                    self.value_params, self.value_state, self.policy_params, self.policy_state, loss, loss_info = self.update(
                        batch, self.value_params, self.value_state, self.policy_params, self.policy_state, learn_policy=learn_policy)
                    if step % log_interval == 0:
                        self.log(loss_info, step=step)
                    step += 1
            rng, sub_rng = split(rng)
            if i % validation_interval == validation_interval - 1:
                mean_cum_reward, best_mean_cum_reward, eval_info = self.eval_step(
                    self.val_env, self.policy_network, self.policy_params, sub_rng, max_episode_length, step,
                    best_mean_cum_reward)
                self.log({"val_cumulative_reward": mean_cum_reward}
                         | eval_info, step)
            self.save_all_params(folder=self.run_dir,
                                 step=step, value=mean_cum_reward, name="last")
        return loss

    def eval_step(self, val_env, policy_network, policy_params, rng, max_episode_length: int, step: int, best_mean_cum_reward):
        mean_cum_reward, eval_info = eval_reward(
            val_env, policy_network, policy_params, seed=rng, max_steps=max_episode_length)
        # self.log({"val_cumulative_reward": mean_cum_reward} | eval_info, step) # logged outside of this function
        if mean_cum_reward >= best_mean_cum_reward:
            best_mean_cum_reward = mean_cum_reward
            self.save_all_params(folder=self.run_dir,
                                 step=step, value=mean_cum_reward, name="best")
        return mean_cum_reward, best_mean_cum_reward, eval_info

    @partial(jit, static_argnums=(0,6))
    def update(self, batched_transition: BatchedPPOTuples, value_params, value_state, policy_params, policy_state, learn_policy=True):
        (loss, loss_info), (policy_grad, value_grad) = value_and_grad(self.loss_fn,
                                                                      argnums=(0, 1), has_aux=True)(policy_params,
                                                                                                    value_params,
                                                                                                    batched_transition)
        policy_updates, policy_state = self.policy_optmizer.update(
            policy_grad, policy_state, policy_params)
        if learn_policy:
            policy_params = optax.apply_updates(policy_params, policy_updates) #TODO :remove this please
        value_updates, value_state = self.value_optimizer.update(
            value_grad, value_state, value_params)
        value_params = optax.apply_updates(value_params, value_updates)
        grad_info = {"policy_grad_norm": optax.global_norm(policy_grad),
                     "value_grad_norm": optax.global_norm(value_grad),
                     "policy_update_norm": optax.global_norm(policy_updates),
                     "value_update_norm": optax.global_norm(value_updates), }
        return value_params, value_state, policy_params, policy_state, loss, loss_info | grad_info

    def mean_ratio_fn(self, policy_params: Dict, batched_transition: BatchedPPOTuples) -> float:
        state = batched_transition.state
        action = batched_transition.action
        prev_log_prob = batched_transition.prev_log_prob
        log_probs, ditrib_params = self.policy_network.vmap_log_prob(
            policy_params, state, action)
        ratio = jnp.exp(log_probs - jax.lax.stop_gradient(prev_log_prob))
        return ratio.mean(), {}

    def loss_fn(self, policy_params: Dict, value_params: Dict, batched_transition: BatchedPPOTuples) -> float:
        # Value loss
        state = batched_transition.state
        value = self.value_network.vmap_apply(value_params, state)
        target_value = jax.lax.stop_gradient(batched_transition.returns)
        # target_value = jnp.ones_like(target_value) # TODO: REMOVE this please
        value_loss = jnp.square(target_value - value).mean()

        # Policy loss
        log_probs, ditrib_params = self.policy_network.vmap_log_prob(
            policy_params, state, batched_transition.action)
        ratio = jnp.exp(
            log_probs - jax.lax.stop_gradient(batched_transition.prev_log_prob))
        advantage = jax.lax.stop_gradient(batched_transition.advantage)
        max_adv = jnp.max(advantage)
        min_adv = jnp.min(advantage)
        # advantage = jnp.clip(advantage, a_min=-0.2, a_max=0.2) # could clip advatanges
        left = ratio * advantage
        right = jnp.clip(ratio, a_min=1 - self.eps,
                         a_max=1+self.eps) * advantage
        policy_loss = -jnp.minimum(left, right).mean()
        clipped_ratio_proportion = (ratio >= (1+self.eps)).mean()

        # Entropy Loss
        entropy = self.policy_network.entropy(ditrib_params, batch=True).mean()
        entropy_loss = - self.alpha_entropy * entropy

        # assert(value.shape == target_value.shape) # OK
        # assert(value.shape == (16,)) # OK
        # assert(target_value.shape == (16,)) # OK
        # assert((target_value - value).shape == (16,)) # OK
        # TODO: verifier si la value loss descend ou pas.
        loss = policy_loss + value_loss + entropy_loss
        abs_value_diff = jnp.abs(value - target_value)
        mean_abs_value_diff = jnp.mean(abs_value_diff)
        return loss, {"loss": loss, "policy_loss": policy_loss, "value_loss": value_loss, "value": value.mean(), "value_target": target_value.mean(), "reward": batched_transition.reward.mean(), 
                      "med_value_diff": jnp.median(abs_value_diff), "max_value_diff": jnp.max(abs_value_diff), "min_value_diff": jnp.min(abs_value_diff), 
                      "mean_value_diff": mean_abs_value_diff, "relative_error": jnp.mean(jnp.abs(value - target_value / (target_value+1e-8))),# TODO,
                      "entropy_loss": entropy_loss, "entropy": entropy,
                      "advantage": advantage.mean(), "max_advantage": max_adv, "min_advantage": min_adv,
                      "ratio_max": ratio.max(), "ratio_min": ratio.min(), "ratio_mean": ratio.mean(),
                      "clipped_ratio_proportion": clipped_ratio_proportion}

    def log(self, metric_dict, step=None):
        self.logger.log_dicts(step, "", metric_dict)

    def sample_batched_episode(self, rng: KeyArray, env: gym.vector.VectorEnv, batch_size,
                               max_episode_length) -> BatchedPPOEpisodes:
        if env.num_envs != batch_size:
            raise ValueError
        return sample_batched_episode(policy=self.policy_network, params=self.policy_params, rng=rng,
                                      deterministic=False, env=env, max_episode_length=max_episode_length)

    def test(self, *, test_env, res_dir, seed=None, max_steps=None) -> float:
        best_value = self._test(test_env, res_dir, name="best",
                   seed=seed, max_steps=max_steps)
        self._test(test_env, res_dir, name="last",
                   seed=seed, max_steps=max_steps)
        
        return best_value
    
    def eval(self, val_env, seed=None, max_steps=None):
        params = self.load_params(name="best", network="policy")
        value, _ = eval_reward(val_env, self.policy_network, params, seed=seed, max_steps=max_steps)
        return value

    def _test(self, test_env, res_dir, name="best", seed=None, max_steps=None) -> float:
        seed = seed or self.seed
        test_name = "test_" + name
        test_dir = os.path.join(res_dir, test_name)
        if not os.path.exists(test_dir):
            os.mkdir(test_dir)
        params = self.load_params(name=name, network="policy")
        value, _ = eval_reward(test_env, self.policy_network, params,seed=seed, save_folder=test_dir, max_steps=max_steps)
        return value

    def _params_name(self, *,  name: str, network: str) -> str:
        return network + "_params_" + name

    def _params_path(self, *, folder, name: str, network: str):
        params_name = self._params_name(name=name, network=network) + ".pkl"
        params_path = os.path.join(folder, params_name)
        return params_path

    def _params_info_path(self, *, folder, name: str, network: str):
        params_info_name = self._params_name(
            name=name, network=network) + "_info.pkl"
        params_info_path = os.path.join(folder, params_info_name)
        return params_info_path

    def save_params(self, folder, params, step=None, value=None, *, name: str, network: str):
        with open(self._params_path(folder=folder, name=name, network=network), 'wb') as f:
            pickle.dump(params, f)
        with open(self._params_info_path(folder=folder, name=name, network=network), 'w') as f:
            json.dump({"step": step, "value": value}, f)

    def save_all_params(self, *, folder, step, value, name):
        assert(name in ["best", "last"])
        self.save_params(folder, self.policy_params, step,
                         value, name=name, network="policy")
        self.save_params(folder, self.value_params, step,
                         value, name=name, network="value")

    def load_params(self, folder=None, *, name: str, network: str) -> dict:
        folder = folder or self.run_dir
        path = self._params_path(folder=folder, name=name, network=network)
        with open(path, 'rb') as f:
            params = pickle.load(f)
        return params

    def save(self, run_dir):
        pass
