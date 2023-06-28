from typing import Dict

import numpy as np
from gymnasium import spaces
from ml4ps.h2mg import (H2MG, H2MGSpace, H2MGStructure, HyperEdgesSpace,
                        HyperEdgesStructure)

from ..VoltageManagement import VoltageManagementState
from .VoltageManagementPandapowerV2 import VoltageManagementPandapowerV2

CONTROL_STRUCTURE = H2MGStructure()
CONTROL_STRUCTURE.add_local_hyper_edges_structure(
    "shunt", HyperEdgesStructure(features=["step"]))

MAX_STEP_STRUCTURE = H2MGStructure()
MAX_STEP_STRUCTURE.add_local_hyper_edges_structure(
    "shunt", HyperEdgesStructure(features=["max_step"]))


# TODO: How to use delta and instantaneous action forlulation with the same environment.
class VoltageManagementPandapowerV2B(VoltageManagementPandapowerV2):
    """Power system environment for voltage management problem implemented
        with pandapower that controls shunt resources.

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
    name = "VoltageManagementPandapowerV2B"
    
    def initialize_control_variables(self, power_grid) -> Dict:
        """Inits control variable with default heuristics."""
        initial_control = self.backend.get_h2mg_from_power_grid(
            power_grid, self.control_structure)
        # infos = self.backend.get_h2mg_from_power_grid(power_grid, self.info_structure)
        # initial_control.flat_array = self.np_random.integers(0, high=infos.local_hyper_edges["shunt"].flat_array,
        # size=initial_control.flat_array.shape)
        initial_control.flat_array = np.zeros_like(initial_control.flat_array)
        return initial_control

