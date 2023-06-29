from typing import Dict, Tuple

import numpy as np
import tqdm
from jax import jit
from jax.random import PRNGKey, split
from ml4ps.logger import dict_mean, mean_of_dicts
from ml4ps.reinforcement.environment import PSBaseEnv, TestEnv
from ml4ps.reinforcement.policy import BasePolicy


# TODO: use eval_reward instead
def test_policy(single_env: PSBaseEnv, policy: BasePolicy, params: Dict, seed, output_dir: str, max_steps=None):
    test_env = TestEnv(single_env, save_folder=output_dir, max_steps=max_steps)
    obs, info = test_env.reset()
    policy_sample = jit(policy.sample, static_argnums=(3,))
    step = 0
    with tqdm.tqdm(total=test_env.maxlen) as pbar:
        while True:
            action, _, _ = policy_sample(params, obs, seed, deterministic=True)
            obs, reward, terminated, truncated, info = test_env.step(action)
            step += 1
            if terminated or truncated:
                pbar.update(1)
            if test_env.is_done:
                break


def eval_reward(single_env: PSBaseEnv, policy: BasePolicy, params: Dict, seed: PRNGKey, n=None, max_steps: int = None, save_folder=None) -> Tuple[float, Dict]:
    if isinstance(seed, int):
        seed = PRNGKey(seed)
    single_env = TestEnv(single_env, auto_reset=False, save_folder=save_folder, max_steps=max_steps)
    obs, info = single_env.reset()
    init_step = single_env.state.power_grid.shunt.step
    # init_setpooints = single_env.state.power_grid.gen.vm_pu
    policy_sample = jit(policy.sample, static_argnums=(3,))
    i = 0
    step = 0
    cumulative_reward = 0
    cumulative_rewards = []
    init_cost = []
    converged = True
    cumulative_rewards_converged = []
    rewards = []
    last_infos = []
    shunt_deltas = []
    stats = {}
    n = min(n, single_env.maxlen) if n is not None else single_env.maxlen
    was_reset = True
    with tqdm.tqdm(total=n, leave=(save_folder is not None)) as pbar:
        while i < n:
            if was_reset:
                init_cost.append(single_env.state.cost)
            seed, val_seed = split(seed)
            # TODO: seed is not used in deterministic TODO change back to deterministic=True
            action, _, _ = policy_sample(
                params, obs, val_seed, deterministic=True)
            obs, reward, terminated, truncated, info = single_env.step(action)
            was_reset = False
            rewards.append(reward)
            converged = converged and (not info["diverged"])
            step += 1
            cumulative_reward += reward
            if terminated or truncated or step >= max_steps:
                shunt_delta = last_action_analysis(
                    single_env.state.power_grid, init_step=init_step)
                shunt_deltas.append(shunt_delta)
                stats[single_env.state.power_grid.name] = cumulative_reward
                last_infos.append(dict_mean(info, nanmean=False))
                cumulative_rewards.append(cumulative_reward)
                if converged:
                    cumulative_rewards_converged.append(cumulative_reward)
                i += 1
                step = 0
                cumulative_reward = 0
                converged = True
                rewards = []
                pbar.update(1)

                # Auto reset for test envs or not ?
                obs, info = single_env.reset(
                    options={"load_new_power_grid": True})
                was_reset = True
                init_step = single_env.state.power_grid.shunt.step

            if single_env.is_done:
                break
    if len(cumulative_rewards) != n:
        raise ValueError(f"{len(cumulative_rewards)} != {n}")

    stats = {"val_" + k: v for (k, v) in stats.items()}

    cumulative_rewards = np.array(cumulative_rewards)
    last_infos_dict = mean_of_dicts(last_infos)
    last_infos_dict = {"val_" + k: v for (k, v) in last_infos_dict.items()}
    mean_cumulative_rewards_converged = np.mean(
        cumulative_rewards_converged) if len(cumulative_rewards_converged) > 0 else 0
    return np.mean(cumulative_rewards), {"cumulative_rewards_converged": mean_cumulative_rewards_converged,
                                         "max_val_cumulative_reward": np.max(cumulative_rewards),
                                         "min_val_cumulative_reward": np.min(cumulative_rewards),
                                         "med_val_cumulative_reward": np.median(cumulative_rewards),
                                         "quant75_val_cumulative_reward": np.quantile(cumulative_rewards, 0.75),
                                         "quant25_val_cumulative_reward": np.quantile(cumulative_rewards, 0.25),
                                         "pos_val_cumulative_reward": np.mean(cumulative_rewards >= 0),
                                         "init_cost": np.mean(init_cost), "max_init_cost": np.max(init_cost),
                                         "min_init_cost": np.min(init_cost), "shunt_deltas": np.mean(shunt_deltas)} | \
        last_infos_dict | stats


def last_action_analysis(power_grid, init_step=None, init_setpoints=None):
    # if init_step is not None:
    #     print("shunt init", init_step)
    #     print("shunt diff", power_grid.shunt.step - init_step)
    #     print("shunt last", power_grid.shunt.step)
    # if init_setpoints is not None:
    #     print("setpoints diff", power_grid.gen.vm_pu - init_setpoints)

    return np.sum(np.abs(power_grid.shunt.step - init_step))
