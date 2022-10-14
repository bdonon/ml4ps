#import julia
#julia.install()


from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, build_unique_id_dict
from julia.api import LibJulia
import os, warnings

api = LibJulia.load()

sysimage = os.getenv('PYJULIA_SYSIMAGE_PATH', None)
if sysimage != None:
    api.sysimage = os.environ['PYJULIA_SYSIMAGE_PATH']
    api.init_julia()
    from julia import Main
    Main.eval('using PowerModels, Ipopt, Gurobi')
else:
	warnings.warn('PowerModels.jl isn''t properly installed.')

VALID_FEATURE_NAMES = {
    'bus': ['zone', 'bus_i', 'bus_type', 'name', 'vmax', 'source_id', 'area', 'vmin', 'index', 'va', 'vm', 'base_kv',
        'lam_kcl_i', 'lam_kcl_r'],
    'load': ['source_id', 'load_bus', 'status', 'qd', 'pd', 'index'],
    'gen': ['ncost', 'qc1max', 'pg', 'model', 'shutdown', 'startup', 'qc2max', 'ramp_agc', 'qg', 'gen_bus', 'pmax',
         'ramp_10', 'vg', 'mbase', 'source_id', 'pc2', 'index', 'cost', 'qmax', 'gen_status', 'qmin', 'qc1min',
         'qc2min', 'pc1', 'ramp_q', 'ramp_30', 'pmin', 'apf', 'qg', 'pg'],
    'shunt': ['source_id', 'shunt_bus', 'status', 'gs', 'bs', 'index'],
    'branch': [ 'br_r', 'rate_a', 'shift', 'br_x', 'g_to', 'g_fr', 'source_id', 'b_fr', 'f_bus', 'br_status',
        't_bus', 'b_to', 'index', 'angmin', 'angmax', 'transformer', 'tap', "qf", "mu_sm_fr", "mu_sm_to", "qt", "pt", "pf"],
     'dcline': [],
     'storage': [],
     'switch': [],
}

VALID_ADDRESS_NAMES = {
    'bus': ['index'],
    'load': ['load_bus'],
    'gen': ['gen_bus'],
    'shunt': ['shunt_bus'],
    'branch': ['f_bus', 't_bus'],
    'dcline': [],
    'storage': [],
    'switch': [],
}


class PowerModelsBackend(AbstractBackend):
    """Backend implementation that uses `PandaPower <http://www.pandapower.org>`_."""

    valid_extensions = (".json", ".m")
    valid_feature_names = VALID_FEATURE_NAMES
    valid_address_names = VALID_ADDRESS_NAMES

    def __init__(self):
        """Initializes a PowerModelsBackend."""
        super().__init__()

    def load_network(self, file_path):
        """Loads a power grid instance"""
        Main.file_path = file_path
        return Main.eval('net = parse_file(file_path)')

    def run_network(self, net, **kwargs):
        """Runs a power flow simulation."""
        return Main.eval('solve_ac_pf')
        
    def run_dc_opf(self, **kwargs):
        """Runs a power flow simulation."""
        return Main.eval('''res = solve_model(net, DCPPowerModel, Gurobi.Optimizer, build_opf,
            setting = Dict("output" => Dict("duals" => true)))''')
        
    def set_feature_network(self, net, y):
        """Updates a power grid by setting features according to `y`."""
        for k in y.keys():
            for f in y[k].keys():
                try:
                    net[k][f] = y[k][f]
                except ValueError:
                    print('Object {} and key {} are not available with PandaPower'.format(k, f))

    def get_feature_network(self, network, feature_names):
        """Returns features from a single power grid instance."""
        print('TODO get_feature_network')

    def get_address_network(self, network, address_names):
        """Extracts a nested dict of address ids from a power grid instance."""
        print('TODO get_address_network')
