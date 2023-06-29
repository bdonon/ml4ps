import os
os.environ["HYDRA_FULL_ERROR"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"

import datetime
import random

import gymnasium as gym
import hydra
from gymnasium.vector import VectorEnv
from omegaconf import OmegaConf

import ml4ps
from ml4ps.logger import log_params_from_omegaconf_dict, get_logger
from ml4ps.reinforcement.algorithm import get_algorithm


OmegaConf.register_new_resolver("add", lambda x, y: x+y)
OmegaConf.register_new_resolver("mul", lambda x, y: x*y)


def get_vector_env(*, name, **kwargs) -> VectorEnv:
    return gym.vector.make(name, **kwargs)


def get_single_env(*, name, **kwargs):
    return gym.make(name, **kwargs)




def save_config(cfg):
    if not os.path.isdir(cfg.res_dir):
        os.mkdir(cfg.res_dir)
    if cfg.run_name is None:
        run_name = f'algo_{random.randint(0,512):03d}_{datetime.datetime.now().strftime("%m%d%Y_%H%M%S")}'
    else:
        run_name = cfg.run_name
    run_dir = os.path.join(cfg.res_dir, run_name)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    return run_dir, run_name


def init_envs(cfg):
    env = get_vector_env(**cfg.env)
    env.reset(seed=cfg.seed)
    val_env = get_single_env(**cfg.val_env)
    val_env.reset(seed=cfg.seed)
    test_env = get_single_env(**cfg.test_env)
    test_env.reset(seed=cfg.seed)
    return env, val_env, test_env


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg):
    # Save configuration
    run_dir, run_name = save_config(cfg)

    # Training environment
    env, val_env, test_env = init_envs(cfg)

    # RL algorithm
    algorithm = get_algorithm(algorithm_type=cfg.algorithm_type, env=env,
                              val_env=val_env, test_env=test_env, run_dir=run_dir, **cfg.algorithm)

    # Logger
    logger = get_logger(**{**cfg.logger, 'run_name': run_name}, run_dir=run_dir)
    logger.log_config(cfg, name="hparam/test", value=-float("inf"))

    # Learning loop
    algorithm.learn(logger=logger, seed=cfg.seed,
                    batch_size=cfg.batch_size, **cfg.learn)
    env.close()
    val_env.close()

    # Save
    algorithm.save(run_dir)

    # Evaluation
    value = algorithm.test(test_env=test_env, res_dir=run_dir, **cfg.test)

    logger.log_config(cfg, name="hparam/test", value=value)

    logger.finalize()

    return value


if __name__ == "__main__":
    main()
