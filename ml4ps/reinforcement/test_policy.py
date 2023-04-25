from typing import Dict
from ml4ps.reinforcement.environment import TestEnv, PSBaseEnv
from ml4ps.reinforcement.policy import BasePolicy
import tqdm
from jax import jit
import numpy as np
from jax.random import PRNGKey, split

def test_policy(single_env: PSBaseEnv, policy: BasePolicy, params: Dict, seed, output_dir: str):
    test_env = TestEnv(single_env, save_folder=output_dir)
    obs, info = test_env.reset()
    policy_sample = jit(policy.sample, static_argnums=(3,))
    with tqdm.tqdm(total=test_env.maxlen) as pbar:
        while True:
            action, _, _ = policy_sample(params, obs, seed, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            if terminated:
                pbar.update(1)
            if test_env.is_done:
                break


def eval_reward(single_env: PSBaseEnv, policy: BasePolicy, params: Dict, seed: PRNGKey, n=100, max_steps: int=20) -> float:
    obs, info = single_env.reset()
    policy_sample = jit(policy.sample, static_argnums=(3,))
    i = 0
    step = 1
    cumulative_reward = 0
    cumulative_rewards = []
    with tqdm.tqdm(total=n, leave=False) as pbar:
        while i < n:
            # seed, val_seed =  split(seed)
            action, _, _ = policy_sample(params, obs, seed, deterministic=True) # TODO: seed is not used in deterministic
            observation, reward, terminated, truncated, info = single_env.step(action)
            step += 1
            cumulative_reward += reward
            if terminated or step >= max_steps:
                cumulative_rewards.append(cumulative_reward)
                i += 1
                step = 0
                cumulative_reward = 0
                pbar.update(1)
                obs, info = single_env.reset(options={"load_new_power_grid": True})
    if len(cumulative_rewards) != n:
        raise ValueError(f"{len(cumulative_rewards)} != {n}")
    return np.mean(cumulative_rewards)
            