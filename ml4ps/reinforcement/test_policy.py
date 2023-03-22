from typing import Dict
from ml4ps.reinforcement.environment import TestEnv, PSBaseEnv
from ml4ps.reinforcement.policy import BasePolicy
import tqdm

def test_policy(single_env: PSBaseEnv, policy: BasePolicy, params: Dict, seed, output_dir: str):
    test_env = TestEnv(single_env, save_folder=output_dir)
    obs, info = test_env.reset()
    with tqdm.tqdm(total=test_env.maxlen) as pbar:
        while True:
            action, _, _ = policy.sample(params, obs, seed, deterministic=True)
            observation, reward, terminated, truncated, info = test_env.step(action)
            if terminated:
                pbar.update(1)
            if test_env.is_done:
                break