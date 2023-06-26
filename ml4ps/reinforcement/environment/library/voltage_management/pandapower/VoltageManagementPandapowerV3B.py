from typing import Dict

import numpy as np
import jax.numpy as jnp
from gymnasium import spaces

from ..VoltageManagement import VoltageManagementState
from .VoltageManagementPandapowerV3 import VoltageManagementPandapowerV3

from ml4ps import h2mg
from ml4ps.h2mg import H2MG, H2MGStructure, H2MGSpace, HyperEdgesStructure, HyperEdgesSpace

CONTROL_STRUCTURE = H2MGStructure()
CONTROL_STRUCTURE.add_local_hyper_edges_structure("gen", HyperEdgesStructure(features=["vm_pu"]))
CONTROL_STRUCTURE.add_local_hyper_edges_structure("ext_grid", HyperEdgesStructure(features=["vm_pu"]))
CONTROL_STRUCTURE.add_local_hyper_edges_structure("shunt", HyperEdgesStructure(features=["step"]))

MAX_STEP_STRUCTURE = H2MGStructure()
MAX_STEP_STRUCTURE.add_local_hyper_edges_structure("shunt", HyperEdgesStructure(features=["max_step"]))

class VoltageManagementPandapowerV3B(VoltageManagementPandapowerV3):
    name = "VoltageManagementPandapowerV3B"
    def initialize_control_variables(self, power_grid) -> Dict:
        """Inits control variable with default heuristics."""
        initial_control = self.backend.get_h2mg_from_power_grid(power_grid, self.control_structure)
        initial_control.local_hyper_edges["gen"].flat_array = jnp.ones_like(initial_control.local_hyper_edges["gen"].flat_array)
        initial_control.local_hyper_edges["ext_grid"].flat_array = jnp.ones_like(initial_control.local_hyper_edges["ext_grid"].flat_array)

        initial_control.local_hyper_edges["shunt"].flat_array = jnp.zeros_like(initial_control.local_hyper_edges["shunt"].flat_array)
        # infos = self.backend.get_h2mg_from_power_grid(power_grid, self.info_structure)
        # initial_control.local_hyper_edges["shunt"].flat_array = self.np_random.integers(0, high=infos.local_hyper_edges["shunt"].flat_array, size=initial_control.local_hyper_edges["shunt"].flat_array.shape)
        return initial_control
