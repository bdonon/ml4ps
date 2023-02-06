from typing import Dict

import numpy as np
from gymnasium import spaces

from ..VoltageManagement import VoltageManagementState
from .VoltageManagementPandapower import VoltageManagementPandapower


class VoltageManagementPandapowerV2(VoltageManagementPandapower):
    """Power system environment for voltage management problem implemented
        with pandapower that controls generators setpoints and shunt resources.

        The setpoints of the action directly override the current setpoints.
        For the shunt steps, the action encodes a variation of step 
            {0: -1 step, 1: no change, 2: +1 step}. The variations are applied and
            the result is clipped in range (0, max_step-1)

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
        self.ctrl_var_names = {"gen": ["vm_pu"],
                               "ext_grid": ["vm_pu"],
                               "shunt": ["step"]}
        self.vlow = 0.8
        self.vhigh = 1.2
        self.action_space = spaces.Dict(
            {"gen": spaces.Dict(
                {"vm_pu": spaces.Box(low=self.vlow, high=self.vhigh, shape=(self.n_obj["gen"],))}),
             "ext_grid": spaces.Dict(
                {"vm_pu": spaces.Box(low=self.vlow, high=self.vhigh, shape=(self.n_obj["ext_grid"],))}),
             "shunt": spaces.Dict(
                {"delta_step": spaces.MultiDiscrete(nvec=np.full(shape=(self.n_obj["shunt"],), fill_value=3))}),
             "stop": spaces.Discrete(2)})

    def initialize_control_variables(self) -> Dict:
        """Inits control variable with default heuristics."""
        n_obj = self.n_obj
        return {"gen": {"vm_pu": np.ones(shape=(n_obj["gen"],), dtype=np.float64)},
                "ext_grid": {"vm_pu": np.ones(shape=(n_obj["ext_grid"],), dtype=np.float64)},
                "shunt": {"step": np.zeros(shape=(n_obj["shunt"],), dtype=np.int32)}}

    def update_ctrl_var(self, ctrl_var: Dict, action: Dict, state: VoltageManagementState) -> Dict:
        """Updates control variables with action."""
        ctrl_var["gen"] = action["gen"]
        ctrl_var["ext_grid"] = action["ext_grid"]
        max_step = self.backend.get_data_power_grid(
            state.power_grid, feature_names={"shunt": ["max_step"]})
        ctrl_var["shunt"]["step"] = np.clip((ctrl_var["shunt"]["step"] + action["shunt"]["delta_step"] - 1),
                                            0, max_step["shunt"]["max_step"])
        return ctrl_var