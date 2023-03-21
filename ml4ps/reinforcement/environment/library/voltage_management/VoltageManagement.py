from abc import ABC, abstractmethod
from collections import namedtuple
from numbers import Number
from typing import Any, Dict, NamedTuple, Optional, Tuple
import os

import numpy as np
from gymnasium import spaces
from ml4ps.reinforcement.environment.ps_environment import PSBaseEnv
from ml4ps.h2mg import H2MG

VoltageManagementState = namedtuple("VoltageSetPointManagementState",
                                    ["power_grid",
                                     "cost",
                                     "iteration",
                                     "stop"])


class VoltageManagement(PSBaseEnv, ABC):
    """Power system environment for voltage management problem.

    Attributes:
        action_space: The Space object corresponding to valid actions.
        address_names: The dict of list of address names for each object class.
        backend: The Backend object that handles power grid manipulation and simulations.
        ctrl_var_names: The dict of list of control variable names for each object class.
        data_dir: The path of the dataset from which power grids will be sampled.
        max_steps: The maximum number of iteration.
        n_obj: The dict of maximum number of objects for each object class in the dataset in data_dir.
        obs_feature_names: The dict with 2 keys "features" and "addresses". "features" contains the dict
            of list of observable features for each object class. "addresses" contains the dict of
            list of observable addresses for eac object class.
        observation_space: the Space object corresponding to valid observations.
        state: The VoltageManagementState named tuple that holds the current state of the environement.
        lmb_i: The float cost hyperparameter corresponding to electric current penalty ponderation.
        lmb_q: The float cost hyperparameter corresponding to reactive reserve penalty ponderation.
        lmb_v: The float cost hyperparameter corresponding to voltage penalty ponderation.
        eps_i: The float cost hyperparameter corresponding to electric current penalty margins.
        eps_q: The float cost hyperparameter corresponding to reactive reserve penalty margins.
        eps_v: The float cost hyperparameter corresponding to voltage penalty margins.
        c_div: The float cost hyperparameter corresponding to the penalty for diverging power grid simulations.
    """
    # Set these in subclasses
    action_space: spaces.Space
    address_names: Dict
    backend: Any
    ctrl_var_names: Dict
    obs_feature_names: Dict

    def __init__(self, data_dir, max_steps=None, cost_hparams=None, soft_reset=True):
        super().__init__(data_dir)
        self.soft_reset = soft_reset
        self.max_steps = max_steps
        self.state = VoltageManagementState(power_grid=None,
                                            cost=None,
                                            iteration=0,
                                            stop=False)

        self.init_cost_hparams(cost_hparams)


    def init_cost_hparams(self, cost_hparams: Dict = None) -> None:
        """Inits cost hparams and overrides default values with given cost_hparams dictionary."""
        if not set(cost_hparams or dict()).issubset(set(self.default_cost_hparams)):
            raise ValueError("Unknown key in cost_hparams")
        cost_hparams = self.default_cost_hparams.update(
            cost_hparams) if cost_hparams is not None else self.default_cost_hparams
        self.lmb_i = cost_hparams["lmb_i"]
        self.lmb_q = cost_hparams["lmb_q"]
        self.lmb_v = cost_hparams["lmb_v"]
        self.eps_i = cost_hparams["eps_i"]
        self.eps_q = cost_hparams["eps_q"]
        self.eps_v = cost_hparams["eps_v"]
        self.c_div = cost_hparams["c_div"]


    def random_power_grid_path(self) -> str:
        """Returns the path of a random power grid in self.data_dir"""
        return self.np_random.choice(self.backend.get_valid_files(self.data_dir))
    
    def get_next_power_grid(self, options=None) -> Any:
        options = options if options is not None else {}
        path = options.get("power_grid_path", self.random_power_grid_path())
        return self.backend.load_power_grid(path)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None) -> Tuple:
        """Resets environment to a new power grid for the given random seed."""
        super().reset(seed=seed)
        power_grid = self.state.power_grid
        if not self.soft_reset or (power_grid is None or options is not None and options.get("load_new_power_grid", False)):
            power_grid = self.get_next_power_grid(options)
        ctrl_var = self.initialize_control_variables(power_grid)
        self.backend.set_h2mg_into_power_grid(power_grid, ctrl_var)
        self.run_power_grid(power_grid)
        iteration = 0
        cost = self.compute_cost(power_grid)
        stop = False
        self.state = VoltageManagementState(power_grid=power_grid,
                                            cost=cost, iteration=iteration, stop=stop)
        obs = self.get_observation(self.state)
        info = self.get_information(state=self.state, action=None)
        return obs, info

    def step(self, action) -> Tuple[H2MG, float, bool, bool, Dict]:
        """Update state with action and return new observation and reward."""
        last_cost = self.state.cost
        self.state = self.dynamics(self.state, action)
        reward = last_cost - self.state.cost
        observation = self.get_observation(self.state)
        terminated = self.is_terminal(self.state)
        truncated = self.is_truncated(self.state)
        info = self.get_information(state=self.state, action=action)
        return observation, reward, terminated, truncated, info

    def dynamics(self, state: NamedTuple, action: Dict) -> NamedTuple:
        """Return new state for a given action"""
        old_ctrl_var = self.backend.get_h2mg_from_power_grid(state.power_grid, structure=self.control_structure)
        ctrl_var = self.update_ctrl_var(old_ctrl_var, action, state)
        self.backend.set_h2mg_into_power_grid(state.power_grid, ctrl_var)
        self.run_power_grid(state.power_grid)
        cost = self.compute_cost(state.power_grid)
        iteration = state.iteration + 1
        stop = self.is_stop(action)
        return VoltageManagementState(state.power_grid, cost, iteration, stop)

    def is_terminal(self, state: NamedTuple) -> bool:
        return state.stop

    def is_truncated(self, state: NamedTuple) -> bool:
        return False

    def is_stop(self, action: H2MG) -> bool:
        if action.global_hyper_edges is not None and "stop" in action.global_hyper_edges.features:
            return bool(action.global_hyper_edges.features["stop"])
        else:
            return True

    def compute_cost(self, power_grid) -> Number:
        """Computes cost of a power grid."""
        if self.has_diverged(power_grid):
            return self.c_div
        else:
            c_i = self.compute_current_cost(power_grid, self.eps_i)
            c_q = self.compute_reactive_cost(power_grid, self.eps_q)
            c_v = self.compute_voltage_cost(power_grid, self.eps_v)
            c_j = self.compute_joule_cost(power_grid)
            return self.lmb_i * c_i + self.lmb_q * c_q + self.lmb_v * c_v + c_j

    def get_feature_names(self, space: spaces.Space) -> Dict:
        """Returns a dict of list of feature names for each object class."""
        return {k: list(v) for k, v in space.items()}

    def get_observation(self, state) -> Dict:
        """Return observation of state wrt self.observation_space."""
        return self.backend.get_h2mg_from_power_grid(state.power_grid, self.observation_structure)

    @property
    def default_cost_hparams(self) -> Dict:
        return {"lmb_i": 1.0, "lmb_q": 1.0, "lmb_v": 1.0, "eps_i": .0, "eps_q": 0.0, "eps_v": 0.0, "c_div": 1.0}
    
    @abstractmethod
    def run_power_grid(self, power_grid):
        self.backend.run_power_grid(power_grid)

    @abstractmethod
    def has_diverged(self, power_grid) -> bool:
        pass

    @abstractmethod
    def initialize_control_variables(self) -> Dict:
        """Inits control variable with default heuristics."""
        pass

    @abstractmethod
    def get_information(self, state, action=None) -> Dict:
        """Gets power grid statistics, cost decomposition, constraints violations and iteration."""
        pass

    @abstractmethod
    def update_ctrl_var(self, ctrl_var: H2MG, action: H2MG, state: VoltageManagementState = None) -> Dict:
        """Updates control variables with action."""
        raise NotImplementedError()

    @abstractmethod
    def compute_current_cost(self, power_grid, eps_i) -> Number:
        """Computes cost corresponding the current limits."""
        pass

    @abstractmethod
    def compute_reactive_cost(self, power_grid, eps_q) -> Number:
        """Computes cost corresponding the reactive power limitation."""
        pass

    @abstractmethod
    def compute_voltage_cost(self, power_grid, eps_v) -> Number:
        """Computes cost corresponding the voltage constraints."""
        pass

    @abstractmethod
    def compute_joule_cost(self, power_grid) -> Number:
        """Computes cost corresponding the joule losses."""
        pass
