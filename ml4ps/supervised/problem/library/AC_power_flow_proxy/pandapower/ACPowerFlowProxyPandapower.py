
from ml4ps.supervised.problem.ps_problem import PSBasePb
from ml4ps.backend import PandaPowerBackend, PaddingWrapper

from gymnasium import spaces
import numpy as np

from ml4ps.reinforcement import H2MGSpace

class ACPowerFlowProxyPandapower(PSBasePb):

    def __init__(self, data_dir, batch_size=1, shuffle=True, load_in_memory=False):
        self.data_dir = data_dir
        self.backend = PaddingWrapper(PandaPowerBackend(), data_dir=self.data_dir)
        self.n_obj = self.backend.max_n_obj.copy()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_in_memory = load_in_memory
        self.global_input_feature_names = {}
        self.local_input_feature_names = {
            "bus": ["in_service", "max_vm_pu", "min_vm_pu", "vn_kv"],
            "load": ["const_i_percent", "const_z_percent", "controllable", "in_service", "p_mw", "q_mvar", "scaling",
                "sn_mva"],
            "sgen": ["controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva", "current_source"],
            "gen": ["controllable", "in_service", "p_mw", "scaling", "sn_mva", "slack", "max_p_mw", "min_p_mw",
                "max_q_mvar", "min_q_mvar", "slack_weight", "vm_pu"],
            "shunt": ["q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service"],
            "ext_grid": ["in_service", "va_degree", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight",
                "vm_pu"],
            "line": ["c_nf_per_km", "df", "g_us_per_km", "in_service", "length_km", "max_i_ka", "max_loading_percent",
                "parallel", "r_ohm_per_km", "x_ohm_per_km"],
            "trafo": ["df", "i0_percent", "in_service", "max_loading_percent", "parallel", "pfe_kw", "shift_degree",
                "sn_mva", "tap_max", "tap_neutral", "tap_min", "tap_phase_shifter", "tap_pos", "tap_side",
                "tap_step_degree", "tap_step_percent", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent"]}
        self.local_address_names = {
            "bus": ["id"], "load": ["bus_id"], "sgen": ["bus_id"], "gen": ["bus_id"], "shunt": ["bus_id"],
            "ext_grid": ["bus_id"], "line": ["from_bus_id", "to_bus_id"], "trafo": ["hv_bus_id", "lv_bus_id"]}
        self.global_output_feature_names = {}
        self.local_output_feature_names = {
            "bus": ["res_vm_pu", "res_va_degree"]} # "res_va_degree"
        self.output_space = H2MGSpace({"local_features":spaces.Dict({"bus": spaces.Dict({"res_vm_pu": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obj["bus"],)),
                                                                                         "res_va_degree": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_obj["bus"],))})})})

        super().__init__()
