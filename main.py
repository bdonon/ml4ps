import datetime
import os

import gymnasium as gym
import hydra
from gymnasium.vector import VectorEnv
from omegaconf import OmegaConf

import ml4ps
from ml4ps.logger import log_params_from_omegaconf_dict
from ml4ps.reinforcement.algorithm import get_algorithm

os.environ["HYDRA_FULL_ERROR"] = "1"

OmegaConf.register_new_resolver("add", lambda x, y: x+y)
OmegaConf.register_new_resolver("mul", lambda x, y: x*y)


def get_vector_env(*, name, **kwargs) -> VectorEnv:
    return gym.vector.make(name, **kwargs)


def get_single_env(*, name, **kwargs):
    return gym.make(name, **kwargs)


def get_logger(*, experiment_name, run_name, **kwargs):
    return ml4ps.logger.MLFlowLogger(experiment_name=experiment_name, run_name=run_name, **kwargs)


def save_config(cfg):
    if not os.path.isdir(cfg.res_dir):
        os.mkdir(cfg.res_dir)
    if cfg.run_name is None:
        run_dir = os.path.join(
            cfg.res_dir, f'algo_{datetime.datetime.now().strftime("%m%d%Y_%H%M%S")}')
    else:
        run_dir = os.path.join(cfg.res_dir, cfg.run_name)
    if not os.path.isdir(run_dir):
        os.mkdir(run_dir)
    with open(os.path.join(run_dir, "config.yaml"), "w") as f:
        OmegaConf.save(cfg, f)
    return run_dir


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
    run_dir = save_config(cfg)

    # Training environment
    env, val_env, test_env = init_envs(cfg)

    # RL algorithm
    algorithm = get_algorithm(algorithm_type=cfg.algorithm_type, env=env,
                              val_env=val_env, test_env=test_env, run_dir=run_dir, **cfg.algorithm)

    # Logger
    logger = get_logger(**cfg.logger)
    log_params_from_omegaconf_dict(cfg)

    # Learning loop
    algorithm.learn(logger=logger, seed=cfg.seed,
                    batch_size=cfg.batch_size, **cfg.learn)
    env.close()
    val_env.close()

    # Save
    algorithm.save(run_dir)

    # Evaluation
    algorithm.test(test_env=test_env, res_dir=run_dir, **cfg.test)

    logger.finalize()


if __name__ == "__main__":
    main()
