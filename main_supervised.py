import torch
import gymnasium as gym
import jax
from hydra import compose, initialize
from hydra.utils import instantiate
from ml4ps.supervised import get_algorithm, get_problem
from ml4ps.logger import MLFlowLogger
import hydra
import os

os.environ["HYDRA_FULL_ERROR"] = "1"


def get_config():
    return {}


def get_logger(*, experiment_name, **kwargs):
    return MLFlowLogger(experiment_name=experiment_name)


@hydra.main(version_base=None, config_path="config", config_name="config_supervised")
def main(cfg):

    # Train, val and test problems
    train_problem = get_problem(problem_type=cfg.train_problem.name, data_dir=cfg.train_problem.data_dir,
                                batch_size=cfg.train_problem.batch_size, shuffle=cfg.train_problem.shuffle,
                                load_in_memory=cfg.train_problem.load_in_memory)
    validation_problem = get_problem(problem_type=cfg.val_problem.name, data_dir=cfg.val_problem.data_dir,
                                     batch_size=cfg.val_problem.batch_size, shuffle=cfg.val_problem.shuffle,
                                     load_in_memory=cfg.val_problem.load_in_memory)
    test_problem = get_problem(problem_type=cfg.test_problem.name, data_dir=cfg.test_problem.data_dir,
                               batch_size=cfg.test_problem.batch_size, shuffle=cfg.test_problem.shuffle,
                               load_in_memory=cfg.test_problem.load_in_memory)

    # Training algorithm
    algorithm = get_algorithm(train_problem=train_problem,
                              validation_problem=validation_problem, test_problem=test_problem, **cfg.algorithm)

    # Logger
    logger = get_logger(**cfg.logger)

    # Learning loop
    algorithm.learn(logger=logger, **cfg.learn)
    # algorithm.test(res_dir=cfg.res_dir)


if __name__ == "__main__":
    main()
