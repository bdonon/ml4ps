from typing import Dict

import numpy as np
import jax.numpy as jnp
from gymnasium import spaces

from ..VoltageManagement import VoltageManagementState
from .VoltageManagementPandapower import VoltageManagementPandapower

from ml4ps import h2mg
from ml4ps.h2mg import H2MG, H2MGStructure, H2MGSpace, HyperEdgesStructure, HyperEdgesSpace

CONTROL_STRUCTURE = H2MGStructure()
CONTROL_STRUCTURE.add_local_hyper_edges_structure("gen", HyperEdgesStructure(features=["vm_pu"]))
CONTROL_STRUCTURE.add_local_hyper_edges_structure("ext_grid", HyperEdgesStructure(features=["vm_pu"]))
CONTROL_STRUCTURE.add_global_hyper_edges_structure(HyperEdgesStructure(features=["stop"]))
CONTROL_STRUCTURE.add_local_hyper_edges_structure("shunt", HyperEdgesStructure(features=["delta_step"]))

class VoltageManagementPandapowerV3(VoltageManagementPandapower):
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
    empty_control_structure = CONTROL_STRUCTURE

    def __init__(self, data_dir, n_obj=None, max_steps=None, cost_hparams=None, additive=True):
        self.additive=additive
        self.name = "VoltageManagementPandapowerV3"
        super().__init__(data_dir, max_steps=max_steps, cost_hparams=cost_hparams)
    
    def _build_action_space(self, control_structure):
        if self.additive:
            offset=0.0
        else:
            offset=1.0
        gen_vm_pu_space = spaces.Box(low=offset-0.2, high=offset+0.2, shape=(control_structure["gen"].features["vm_pu"],))
        ext_grid_vm_pu_space = spaces.Box(low=offset-0.2, high=offset+0.2, shape=(control_structure["ext_grid"].features["vm_pu"],))

        action_space = H2MGSpace()
        action_space._add_hyper_edges_space('gen', HyperEdgesSpace(features=spaces.Dict({"vm_pu": gen_vm_pu_space})))
        action_space._add_hyper_edges_space('ext_grid', HyperEdgesSpace(features=spaces.Dict(
            {"vm_pu": ext_grid_vm_pu_space})))

        delta_step_space = spaces.MultiDiscrete(nvec=np.full(shape=(control_structure["ext_grid"].features["step"],), fill_value=3))
        action_space._add_hyper_edges_space("shunt", HyperEdgesSpace(features=spaces.Dict({"delta_step": delta_step_space})))

        stop_space = spaces.Discrete(2)
        action_space._add_hyper_edges_space("global", HyperEdgesSpace(features=spaces.Dict({"stop": stop_space})))
        
        return action_space

    def initialize_control_variables(self, power_grid, initial_control: H2MG) -> Dict:
        """Inits control variable with default heuristics."""
        initial_control = self.backend.get_h2mg_from_power_grid(power_grid, self.control_structure)
        initial_control.flat_array = 1. + 0. * initial_control.flat_array
        initial_control.local_hyper_edges["gen"].flat_array = jnp.ones_like(initial_control.local_hyper_edges["gen"].flat_array)
        initial_control.local_hyper_edges["shunt"].flat_array = jnp.zeros_like(initial_control.local_hyper_edges["shunt"].flat_array)
        return initial_control

    def update_ctrl_var(self, ctrl_var: Dict, action: Dict, state: VoltageManagementState) -> Dict:
        """Updates control variables with action."""
        # ctrl_var["gen"] = action["gen"]
        # ctrl_var["ext_grid"] = action["ext_grid"]
        max_step = self.backend.get_data_power_grid(
            state.power_grid, feature_names={"shunt": ["max_step"]})
        max_step = h2mg.local_features(max_step)
        # ctrl_var["shunt"]["step"] = np.clip((ctrl_var["shunt"]["step"] + action["shunt"]["delta_step"] - 1),
        #                                     0, max_step["shunt"]["max_step"])
        res = H2MG.from_structure(ctrl_var.structure)
        res.local_hyper_edges["shunt"].flat_array = res.local_hyper_edges["shunt"].flat_array + action.local_hyper_edges["shunt"].flat_array
        res.local_hyper_edges["shunt"].flat_array = np.clip(res.local_hyper_edges["shunt"].flat_array, 0, max_step)
        if self.additive:
            res.local_hyper_edges["gen"] = res.local_hyper_edges["gen"] + action.local_hyper_edges["gen"]
        else:
            res.local_hyper_edges["gen"] = action.local_hyper_edges["gen"]
        return res
