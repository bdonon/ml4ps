from typing import Dict

import numpy as np
from gymnasium import spaces

from ..VoltageManagement import VoltageManagementState
from .VoltageManagementPandapower import VoltageManagementPandapower


class VoltageManagementPandapowerV1(VoltageManagementPandapower):
    """Power system environment for voltage management problem implemented
        with pandapower that controls generators setpoints.

        The action directly overrides the setpoints of the generators.

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

    def __init__(self, data_dir, n_obj=None, max_steps=None, cost_hparams=None):
        super().__init__(data_dir, n_obj=n_obj, max_steps=max_steps, cost_hparams=cost_hparams)
        self.vlow = 0.8
        self.vhigh = 1.2
        self.ctrl_var_names = {"gen": ["vm_pu"],
                               "ext_grid": ["vm_pu"]}
        self.action_space = spaces.Dict({"local_features":
            spaces.Dict({"gen":      spaces.Dict({"vm_pu":
                                      spaces.Box(low=self.vlow,
                                                 high=self.vhigh,
                                                 shape=(self.n_obj["gen"],))}),
             "ext_grid": spaces.Dict({"vm_pu":
                                      spaces.Box(low=self.vlow,
                                                 high=self.vhigh,
                                                 shape=(self.n_obj["ext_grid"],))})})})

    def initialize_control_variables(self) -> Dict:
        """Inits control variable with default heuristics."""
        return {"local_features":{"gen": {"vm_pu": np.ones(shape=(self.n_obj["gen"],), dtype=np.float64)},
                "ext_grid": {"vm_pu": np.ones(shape=(self.n_obj["ext_grid"],), dtype=np.float64)}}}

    def update_ctrl_var(self, ctrl_var: Dict, action: Dict, state: VoltageManagementState) -> Dict:
        """Updates control variables with action."""
        return action
    
    def run_power_grid(self, power_grid):
        return self.backend.run_power_grid(power_grid, enforce_q_lims=True, delta_q=0.,
                                           recycle={"bus_pq":False, "gen":True, "trafo": False})
