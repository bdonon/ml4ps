
from ml4ps.supervised.problem.ps_problem import PSBasePb
from ml4ps.backend import PandaPowerBackend, PaddingWrapper

from gymnasium import spaces
import numpy as np

from ml4ps.h2mg import H2MGSpace, H2MGStructure, HyperEdgesStructure, HyperEdgesSpace


INPUT_STRUCTURE = H2MGStructure()

bus_structure = HyperEdgesStructure(addresses=["id"], features=["in_service", "max_vm_pu", 'min_vm_pu', "vn_kv"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("bus", bus_structure)

load_structure = HyperEdgesStructure(addresses=["bus_id"], features=["const_i_percent", "const_z_percent",
    "controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("load", load_structure)

sgen_structure = HyperEdgesStructure(addresses=["bus_id"], features=["in_service", "p_mw", "q_mvar", "scaling",
    "sn_mva", "current_source"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("sgen", sgen_structure)

gen_structure = HyperEdgesStructure(addresses=["bus_id"], features=["controllable", "in_service", "p_mw",
    "scaling", "sn_mva", "vm_pu", "slack", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("gen", gen_structure)

shunt_structure = HyperEdgesStructure(addresses=["bus_id"], features=["q_mvar", "p_mw", "vn_kv", "step",
    "max_step", "in_service"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("shunt", shunt_structure)

ext_grid_structure = HyperEdgesStructure(addresses=["bus_id"], features=["in_service", "va_degree", "vm_pu",
    "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar", "slack_weight"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("ext_grid", ext_grid_structure)

line_structure = HyperEdgesStructure(addresses=["from_bus_id", "to_bus_id"], features=["c_nf_per_km", "df",
    "g_us_per_km", "in_service", "length_km", "max_i_ka", "max_loading_percent", "parallel", "r_ohm_per_km",
    "x_ohm_per_km"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("line", line_structure)

trafo_structure = HyperEdgesStructure(addresses=["hv_bus_id", "lv_bus_id"], features=["df", "i0_percent", "in_service",
    "max_loading_percent", "parallel", "pfe_kw", "shift_degree", "sn_mva", "tap_max", "tap_neutral", "tap_min",
    "tap_phase_shifter", "tap_pos", "tap_side", "tap_step_degree", "tap_step_percent", "vn_hv_kv", "vn_lv_kv",
    "vk_percent", "vkr_percent"])
INPUT_STRUCTURE.add_local_hyper_edges_structure("trafo", trafo_structure)


OUTPUT_STRUCTURE = H2MGStructure()

bus_structure = HyperEdgesStructure(addresses=["id"], features=["res_vm_pu", "res_va_degree"])
OUTPUT_STRUCTURE.add_local_hyper_edges_structure("bus", bus_structure)


class ACPowerFlowProxyPandapower(PSBasePb):

    backend = PaddingWrapper(PandaPowerBackend())
    empty_input_structure = INPUT_STRUCTURE
    empty_output_structure = OUTPUT_STRUCTURE

    def __init__(self, data_dir: str, batch_size: int = 8, shuffle: bool = True, load_in_memory: bool = True):
        super().__init__(data_dir=data_dir, batch_size=batch_size, shuffle=shuffle, load_in_memory=load_in_memory)

    def _build_output_space(self, output_structure):
        gen_vm_pu_space = spaces.Box(low=0.8, high=1.2, shape=(output_structure["bus"].features["res_vm_pu"],))
        gen_va_degree_space = spaces.Box(low=0.8, high=1.2, shape=(output_structure["bus"].features["res_va_degree"],))

        action_space = H2MGSpace()
        action_space._add_hyper_edges_space('bus',
            HyperEdgesSpace(features=spaces.Dict({"res_vm_pu": gen_vm_pu_space, "res_va_degree": gen_va_degree_space})))
        return action_space
