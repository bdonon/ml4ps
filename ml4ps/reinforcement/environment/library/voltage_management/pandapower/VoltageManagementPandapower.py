from abc import abstractmethod
from numbers import Number
from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces
from ml4ps import PaddingWrapper, PandaPowerBackend
from ml4ps.h2mg import H2MG, H2MGStructure, HyperEdgesStructure

from ..VoltageManagement import VoltageManagement, VoltageManagementState


OBSERVATION_STRUCTURE = H2MGStructure()

bus_structure = HyperEdgesStructure(addresses=["id"], features=["in_service", "max_vm_pu", 'min_vm_pu', "vn_kv"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("bus", bus_structure)

load_structure = HyperEdgesStructure(addresses=["bus_id"], features=["const_i_percent", "const_z_percent",
    "controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("load", load_structure)

sgen_structure = HyperEdgesStructure(addresses=["bus_id"], features=["in_service", "p_mw", "q_mvar", "scaling",
    "sn_mva", "current_source"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("sgen", sgen_structure)

gen_structure = HyperEdgesStructure(addresses=["bus_id"], features=["controllable", "in_service", "p_mw",
    "scaling", "sn_mva", "vm_pu", "slack", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("gen", gen_structure)

shunt_structure = HyperEdgesStructure(addresses=["bus_id"], features=["q_mvar", "p_mw", "vn_kv", "step",
    "max_step", "in_service"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("shunt", shunt_structure)

ext_grid_structure = HyperEdgesStructure(addresses=["bus_id"], features=["in_service", "va_degree", "vm_pu",
    "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("ext_grid", ext_grid_structure)

line_structure = HyperEdgesStructure(addresses=["from_bus_id", "to_bus_id"], features=["c_nf_per_km", "df",
    "g_us_per_km", "in_service", "length_km", "max_i_ka", "max_loading_percent", "parallel", "r_ohm_per_km",
    "x_ohm_per_km"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("line", line_structure)

trafo_structure = HyperEdgesStructure(addresses=["hv_bus_id", "lv_bus_id"], features=["df", "i0_percent", "in_service",
    "max_loading_percent", "parallel", "pfe_kw", "shift_degree", "sn_mva", "tap_max", "tap_neutral", "tap_min",
    "tap_phase_shifter", "tap_pos", "tap_side", "tap_step_degree", "tap_step_percent", "vn_hv_kv", "vn_lv_kv",
    "vk_percent", "vkr_percent"])
OBSERVATION_STRUCTURE.add_local_hyper_edges_structure("trafo", trafo_structure)


class VoltageManagementPandapower(VoltageManagement):
    """Power system environment for voltage management problem implemented with pandapower

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
    backend = PaddingWrapper(PandaPowerBackend())
    empty_observation_structure = OBSERVATION_STRUCTURE


    def __init__(self, data_dir, max_steps=None, cost_hparams=None, soft_reset=True):
        super().__init__(data_dir, max_steps=max_steps, cost_hparams=cost_hparams, soft_reset=soft_reset)

    def has_diverged(self, power_grid) -> bool:
        return not power_grid.converged

    def compute_current_cost(self, power_grid, eps_i) -> Number:
        line_loading_percent = power_grid.res_line.loading_percent
        transfo_loading_percent = power_grid.res_trafo.loading_percent
        loading_percent = np.concatenate([line_loading_percent, transfo_loading_percent], axis=-1)
        return self.normalized_cost(loading_percent, 0, 100, 0, 2*eps_i)

    def compute_reactive_cost(self, power_grid, eps_q) -> Number:
        q = np.concatenate([power_grid.res_gen.q_mvar, power_grid.res_ext_grid.q_mvar], axis=-1)
        qmin = np.concatenate([power_grid.gen.min_q_mvar, power_grid.ext_grid.min_q_mvar], axis=-1)
        qmax = np.concatenate([power_grid.gen.max_q_mvar, power_grid.ext_grid.max_q_mvar], axis=-1)
        return self.normalized_cost(q, qmin, qmax, eps_q, eps_q)

    def compute_voltage_cost(self, power_grid, eps_v) -> Number:
        v = power_grid.res_bus.vm_pu
        vmin = power_grid.bus.min_vm_pu
        vmax = power_grid.bus.max_vm_pu
        return self.normalized_cost(v, vmin, vmax, eps_v, eps_v)

    def compute_joule_cost(self, power_grid) -> Number:
        line_losses = power_grid.res_line.pl_mw
        transfo_losses = power_grid.res_trafo.pl_mw
        loads = power_grid.res_load.p_mw
        joule_losses = np.nansum(line_losses) + np.nansum(transfo_losses)
        total_load = np.nansum(loads)
        return joule_losses / total_load

    def normalized_cost(self, value, min_value, max_value, eps_min_threshold, eps_max_threshold)  -> Number:
        v = (value - min_value) / (max_value - min_value)
        # TODO: change loss
        return np.nanmean(np.greater(v, 1-eps_max_threshold) * np.power(v - (1-eps_max_threshold), 2)
                + np.greater(eps_min_threshold, v) * np.power(eps_min_threshold - v, 2))

    def get_information(self, state: VoltageManagementState, action: Dict = None, reward: float = None) -> Dict:
        """Gets power grid statistics, cost decomposition, constraints violations and iteration."""
        power_grid = state.power_grid
        if not self.has_diverged(power_grid):
            cost = self.compute_cost(power_grid)
            c_i = self.compute_current_cost(power_grid, self.eps_i)
            c_q = self.compute_reactive_cost(power_grid, self.eps_q)
            c_v = self.compute_voltage_cost(power_grid, self.eps_v)
            c_j = self.compute_joule_cost(power_grid)
            is_violated_dict, violated_percentage_dict = self.compute_constraint_violation(power_grid)

            info = {"diverged": self.has_diverged(power_grid),
                "cost": cost, "c_i": c_i, "c_q": c_q, "c_v": c_v, "c_j": c_j, "iteration": state.iteration,
                **is_violated_dict, **violated_percentage_dict}
            if reward is not None:
                reward_info = {"reward": reward, "pos_reward": reward >= 0}
                info = info | reward_info
            return info
        else:
            return {"diverged": self.has_diverged(power_grid)}

    def compute_constraint_violation(self, power_grid) -> Tuple[Dict, Dict]:
        """Computes constraints violation statistics

        Args:
            power_grid (obj): pandapower power grid object

        Returns:
            Tuple[Dict, Dict]: 2 dictionaries of constraints violation.
            The first one constains bool values indicating whether or not
            there was a constraint violation in the power grid.
            The second one indicates the percentage of constraint violation 
            w.r.t the number of objects.
        """
        voltage_violated_bus = np.logical_or(power_grid.res_bus.vm_pu > power_grid.bus.max_vm_pu,
                                             power_grid.res_bus.vm_pu < power_grid.bus.min_vm_pu)
        voltage_violated_percentage = voltage_violated_bus.mean()
        voltage_violated = voltage_violated_bus.any().astype(int)
        loading_connexion = np.concatenate(
            [power_grid.res_line.loading_percent, power_grid.res_trafo.loading_percent], axis=-1)
        loading_violated_connexion = loading_connexion > 100
        loading_violated_percentage = loading_violated_connexion.mean()
        loading_violated = loading_violated_connexion.any().astype(int)
        q = np.concatenate([power_grid.res_gen.q_mvar, power_grid.res_ext_grid.q_mvar], axis=-1)
        qmin = np.concatenate([power_grid.gen.min_q_mvar, power_grid.ext_grid.min_q_mvar], axis=-1)
        qmax = np.concatenate([power_grid.gen.max_q_mvar, power_grid.ext_grid.max_q_mvar], axis=-1)
        reactive_power_violated_gen = np.logical_or(q > qmax, q < qmin)
        reactive_power_violated_percentage = reactive_power_violated_gen.mean()
        reactive_power_violated = reactive_power_violated_gen.any().astype(int)
        return {"voltage_violated": voltage_violated,
                "loading_violated": loading_violated,
                "reactive_power_violated": reactive_power_violated}, \
               {"voltage_violated_percentage": voltage_violated_percentage * 100,
               "loading_violated_percentage": loading_violated_percentage * 100,
               "reactive_power_violated_percentage": reactive_power_violated_percentage * 100}, \
    
    def run_power_grid(self, power_grid):
        return self.backend.run_power_grid(power_grid, enforce_q_lims=True, delta_q=0.)

    @abstractmethod
    def update_ctrl_var(self, ctrl_var: H2MG, action: H2MG) -> Dict:
        """Updates control variables with action."""
        pass

    @abstractmethod
    def initialize_control_variables(self, power_grid) -> Dict:
        """Inits control variable with default heuristics."""
        pass
