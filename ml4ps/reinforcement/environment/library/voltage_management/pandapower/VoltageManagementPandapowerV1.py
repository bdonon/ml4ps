from typing import Dict

from gymnasium import spaces

from ..VoltageManagement import VoltageManagementState
from .VoltageManagementPandapower import VoltageManagementPandapower
from ml4ps.h2mg import H2MG, H2MGStructure, H2MGSpace, HyperEdgesStructure, HyperEdgesSpace


CONTROL_STRUCTURE = H2MGStructure()
CONTROL_STRUCTURE.add_local_hyper_edges_structure("gen", HyperEdgesStructure(features=["vm_pu"]))
CONTROL_STRUCTURE.add_local_hyper_edges_structure("ext_grid", HyperEdgesStructure(features=["vm_pu"]))


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
    empty_control_structure = CONTROL_STRUCTURE

    def __init__(self, data_dir, max_steps=None, cost_hparams=None, soft_reset=True, additive=True):
        self.name = "VoltageManagementPandapowerV1"
        self.additive = additive
        super().__init__(data_dir, max_steps=max_steps, cost_hparams=cost_hparams, soft_reset=soft_reset)

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
        return action_space

    def initialize_control_variables(self, power_grid) -> Dict:
        """Inits control variable with default heuristics."""
        initial_control = self.backend.get_h2mg_from_power_grid(power_grid, self.control_structure)
        initial_control.flat_array = 1. + 0. * initial_control.flat_array
        return initial_control

    def update_ctrl_var(self, ctrl_var: H2MG, action: H2MG, state: VoltageManagementState) -> Dict:
        """Updates control variables with action."""
        if self.additive:
            res = H2MG.from_structure(ctrl_var.structure)
            res.flat_array = ctrl_var.flat_array + action.flat_array
            return res
        else:
            return action
    
    def run_power_grid(self, power_grid):
        return self.backend.run_power_grid(power_grid, enforce_q_lims=True, delta_q=0.,
                                           recycle={"bus_pq":False, "gen":True, "trafo": False})
