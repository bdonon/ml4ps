import torch
import ml4ps
import gymnasium as gym
import ml4ps
import jax
from hydra import compose, initialize
from hydra.utils import instantiate
import hydra
import os
os.environ["HYDRA_FULL_ERROR"]="1"

def get_vector_env(*, env_name, num_envs, data_dir, **kwargs):
    return gym.vector.make(env_name, data_dir=data_dir, num_envs=num_envs, soft_reset=False)

def get_single_env(*, env_name, data_dir, **kwargs):
    return gym.make(env_name, data_dir=data_dir)

def get_config():
    return {}

def get_logger(*, experiment_name, **kwargs):
    return ml4ps.logger.MLFlowLogger(experiment_name=experiment_name)

def get_algorithm(*, env, policy_type, **kwargs):
    # ml4ps.reinforcement.algorithm.get(env, **kwargs)
    return ml4ps.reinforcement.algorithm.Reinforce(env, policy_type=policy_type)

def instantiate_algorithm(algo_cfg):
    return instantiate(algo_cfg)

def get_algorithm(*, env, policy_type, **kwargs):
    # ml4ps.reinforcement.algorithm.get(env, **kwargs)
    return ml4ps.reinforcement.algorithm.Reinforce(env, policy_type=policy_type, **kwargs)

@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Training environment
    env = get_vector_env(env_name=cfg.env.name, data_dir =cfg.env.data_dir, num_envs=cfg.env.num_envs, train=True)

    # RL algorithm
    single_env = get_single_env(env_name=cfg.env.name, data_dir=cfg.env.data_dir, train=True)
    single_obs = single_env.reset()[0]
    algorithm = get_algorithm(env=env, init_obs=single_obs, **cfg.algorithm)


    # Logger
    logger = get_logger(**cfg.logger)
    
    # Learning loop
    algorithm.learn(logger=logger, seed=cfg.seed, batch_size=cfg.batch_size, **cfg.learn)

    # Evaluation loop
    # algorithm.eval(logger=logger, seed=cfg.seed, batch_size=cfg.batch_size, **cfg.learn)

if __name__ == "__main__":
    main()
