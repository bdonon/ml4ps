from abc import abstractmethod
from numbers import Number
from typing import Any, Dict, Tuple

import numpy as np
from gymnasium import spaces
from ml4ps import PaddingWrapper, PandaPowerBackend

from ..VoltageManagement import VoltageManagement, VoltageManagementState


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
    # Set these in subclasses
    action_space: spaces.Space
    address_names: Dict
    backend: Any
    ctrl_var_names: Dict
    obs_feature_names: Dict

    def __init__(self, data_dir, n_obj=None, max_steps=None, cost_hparams=None):
        self.address_names = {
            "bus": ["id"],
            "load": ["bus_id"],
            "sgen": ["bus_id"],
            "gen": ["bus_id"],
            "shunt": ["bus_id"],
            "ext_grid": ["bus_id"],
            "line": ["from_bus_id", "to_bus_id"],
            "trafo": ["hv_bus_id", "lv_bus_id"]
        }
        self.obs_feature_names = {
            "bus": ["in_service", "max_vm_pu", "min_vm_pu", "vn_kv"],
            "load": ["const_i_percent", "const_z_percent", "controllable", "in_service",
                     "p_mw", "q_mvar", "scaling", "sn_mva"],
            "sgen": ["controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva",
                     "current_source"],
            "gen": ["controllable", "in_service", "p_mw", "scaling", "sn_mva",
                    "slack", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar",
                    "slack_weight"],
            "shunt": ["q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"],
            "ext_grid": ["in_service", "va_degree", "max_p_mw", "min_p_mw", "max_q_mvar",
                         "min_q_mvar", "slack_weight"],
            "line": ["c_nf_per_km", "df", "g_us_per_km", "in_service", "length_km", "max_i_ka",
                     "max_loading_percent", "parallel", "r_ohm_per_km", "x_ohm_per_km"],
            "trafo": ["df", "i0_percent", "in_service", "max_loading_percent", "parallel",
                      "pfe_kw", "shift_degree", "sn_mva", "tap_max", "tap_neutral", "tap_min",
                      "tap_phase_shifter", "tap_pos", "tap_side", "tap_step_degree",
                      "tap_step_percent", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent"],
        }
        self.backend = PaddingWrapper(PandaPowerBackend(), data_dir=data_dir)
        super().__init__(data_dir, address_names=self.address_names, obs_feature_names=self.obs_feature_names,
                         n_obj=n_obj, max_steps=max_steps, cost_hparams=cost_hparams)

    def has_diverged(self, power_grid) -> bool:
        return not power_grid.converged

    def compute_current_cost(self, power_grid, eps_i) -> Number:
        data = self.backend.get_data_power_grid(power_grid, feature_names={"line":  ["res_loading_percent"],
                                                                           "trafo": ["res_loading_percent"]})
        line_loading_percent = data["line"]["res_loading_percent"]
        transfo_loading_percent = data["trafo"]["res_loading_percent"]
        loading_percent = np.concatenate([line_loading_percent, transfo_loading_percent], axis=-1)
        return self.normalized_cost(loading_percent, 0, 100, 0, eps_i)

    def compute_reactive_cost(self, power_grid, eps_q) -> Number:
        data = self.backend.get_data_power_grid(power_grid,
                                feature_names={"gen":  ["max_q_mvar", "min_q_mvar", "res_q_mvar"],
                                               "ext_grid":  ["max_q_mvar", "min_q_mvar", "res_q_mvar"]})
        q = np.concatenate(
            [data["gen"]["res_q_mvar"], data["ext_grid"]["res_q_mvar"]], axis=-1)
        qmin = np.concatenate(
            [data["gen"]["min_q_mvar"], data["ext_grid"]["min_q_mvar"]], axis=-1)
        qmax = np.concatenate(
            [data["gen"]["max_q_mvar"], data["ext_grid"]["max_q_mvar"]], axis=-1)
        return self.normalized_cost(q, qmin, qmax, eps_q, eps_q)

    def compute_voltage_cost(self, power_grid, eps_v) -> Number:
        data = self.backend.get_data_power_grid(
            power_grid, feature_names={"bus": ["res_vm_pu", "max_vm_pu", "min_vm_pu"]})
        v = data["bus"]["res_vm_pu"]
        vmin = data["bus"]["min_vm_pu"]
        vmax = data["bus"]["max_vm_pu"]
        return self.normalized_cost(v, vmin, vmax, eps_v, eps_v)

    def compute_joule_cost(self, power_grid)  -> Number:
        data = self.backend.get_data_power_grid(power_grid,
                                                feature_names={"line":  ["res_pl_mw"],
                                                               "trafo": ["res_pl_mw"], "load": ["res_p_mw"]})
        line_losses = data["line"]["res_pl_mw"]
        transfo_losses = data["trafo"]["res_pl_mw"]
        loads = data["load"]["res_p_mw"]
        joule_losses = line_losses.sum() + transfo_losses.sum()
        total_load = np.where(np.isnan(loads), 1e-8, loads).sum()
        return joule_losses / total_load

    def normalized_cost(self, value, min_value, max_value, eps_min_threshold, eps_max_threshold)  -> Number:
        v = (value - min_value) / (max_value - min_value)
        return (np.greater(v, 1-eps_max_threshold) * np.power(v - (1-eps_max_threshold), 2)
                + np.greater(eps_min_threshold, v) * np.power(eps_min_threshold - v, 2)).mean()

    def get_information(self, state: VoltageManagementState, action: Dict = None) -> Dict:
        """Gets power grid statistics, cost decomposition, constraints violations and iteration."""
        power_grid = self.state.power_grid
        cost = self.compute_cost(power_grid)
        c_i = self.compute_current_cost(power_grid, self.eps_i)
        c_q = self.compute_reactive_cost(power_grid, self.eps_q)
        c_v = self.compute_voltage_cost(power_grid, self.eps_v)
        c_j = self.compute_joule_cost(power_grid)
        is_violated_dict, violated_percentage_dict = self.compute_constraint_violation(power_grid)
        return {"diverged": self.has_diverged(self.state.power_grid),
                "cost": cost, "c_i": c_i, "c_q": c_q, "c_v": c_v, "c_j": c_j,
                "action": action, "iteration": self.state.iteration,
                **is_violated_dict, **violated_percentage_dict}

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
        data = self.backend.get_data_power_grid(power_grid,
                                                feature_names={
                                                    "global": ["converged"],
                                                    "bus": ["res_vm_pu", "max_vm_pu", "min_vm_pu"],
                                                    "line":  ["res_pl_mw", "res_loading_percent"],
                                                    "trafo": ["res_pl_mw", "res_loading_percent"],
                                                    "gen":  ["max_q_mvar", "min_q_mvar", "res_q_mvar"],
                                                    "ext_grid":  ["max_q_mvar", "min_q_mvar", "res_q_mvar"],
                                                    "load": ["res_p_mw"]})
        voltage_violated_bus = np.logical_or(data["bus"]["res_vm_pu"] > data["bus"]["max_vm_pu"],
                                             data["bus"]["res_vm_pu"] < data["bus"]["min_vm_pu"])
        voltage_violated_percentage = voltage_violated_bus.mean()
        voltage_violated = voltage_violated_bus.any()
        loading_violated_connexion = np.concatenate(
            [data["line"]["res_loading_percent"], data["trafo"]["res_loading_percent"]], axis=-1)
        loading_violated_percentage = loading_violated_connexion.mean()
        loading_violated = loading_violated_connexion.any()
        q = np.concatenate(
            [data["gen"]["res_q_mvar"], data["ext_grid"]["res_q_mvar"]], axis=-1)
        qmin = np.concatenate(
            [data["gen"]["min_q_mvar"], data["ext_grid"]["min_q_mvar"]], axis=-1)
        qmax = np.concatenate(
            [data["gen"]["max_q_mvar"], data["ext_grid"]["max_q_mvar"]], axis=-1)
        reactive_power_violated_gen = np.logical_or(q > qmax, q < qmin)
        reactive_power_violated_percentage = reactive_power_violated_gen.mean()
        reactive_power_violated = reactive_power_violated_gen.any()
        return {"voltage_violated": voltage_violated,
                "loading_violated": loading_violated,
                "reactive_power_violated": reactive_power_violated}, \
               {"voltage_violated_percentage": voltage_violated_percentage,
               "loading_violated_percentage": loading_violated_percentage,
               "reactive_power_violated_percentage": reactive_power_violated_percentage}

    @abstractmethod
    def update_ctrl_var(self, ctrl_var: Dict, action: Dict) -> Dict:
        """Updates control variables with action."""
        pass

    @abstractmethod
    def initialize_control_variables(self) -> Dict:
        """Inits control variable with default heuristics."""
        pass
