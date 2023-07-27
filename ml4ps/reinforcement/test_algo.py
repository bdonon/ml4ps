from typing import Dict, Tuple

from ml4ps.reinforcement.algorithm import Algorithm
from ml4ps.reinforcement.policy import get_policy
from .test_policy import eval_reward

from omegaconf import OmegaConf

import os
import pickle
import gymnasium as gym
import json


def test_algo(folder, save_folder: str, algorithm: Algorithm=None, test_env_name=None, test_env_dir=None, test_env_args=None):
        if algorithm is None:
            best_params_name = Algorithm.best_params_name
            best_params_info_name = Algorithm.best_params_info_name
        else:
            best_params_name = algorithm.best_params_name
            best_params_info_name = algorithm.best_params_info_name
        
        cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
        if test_env_name is None:
            test_env_name = cfg.test_env.name
        if test_env_dir is None:
            test_env_dir = cfg.test_env.data_dir
        if test_env_args is None:
            test_env_args = {}
        test_env = get_single_env(**{**cfg.test_env, "name":test_env_name, "data_dir":test_env_dir, **test_env_args})
        train_env = get_vector_env(**cfg.env)
        print(f"testing on env {test_env_name} from {test_env_dir}")
        best_params_path = os.path.join(folder, best_params_name)
        with open(best_params_path, 'rb') as f:
            params = pickle.load(f)
        with open(os.path.join(folder, best_params_info_name), 'r') as f:
            params_info = json.load(f)
            print("Best params info:", params_info)
        train_policy = get_policy(cfg.algorithm.policy_type, train_env, **cfg.algorithm.policy_args, nn_args=cfg.algorithm.nn_args)
        policy = get_policy(cfg.algorithm.policy_type, test_env, **cfg.algorithm.policy_args, nn_args=cfg.algorithm.nn_args, normalizer=train_policy.normalizer)
        train_env.close()
        save_path = os.path.join(folder, save_folder)
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        value, eval_info = eval_reward(test_env, policy, params, seed=0, max_steps=cfg.max_steps, save_folder=save_path)
        test_env.close()
        print("Value: ", value)

        with open(os.path.join(save_path, 'info.json'), 'w') as f:
            json.dump({"training_step": params_info.get("step"), "validation_value": params_info.get("value"),
                            "test_value": value, **eval_info}, f)


def get_single_env(*, name, **kwargs):
    return gym.make(name, **kwargs)

def get_vector_env(*, name, **kwargs):
    return gym.vector.make(name, **kwargs)