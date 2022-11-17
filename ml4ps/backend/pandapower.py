import pandas as pd

from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, convert_addresses_to_integers
import pandapower as pp
import pandapower.converter as pc
import mat73
import numpy as np
import os
from types import SimpleNamespace

from pandapower.converter.matpower.from_mpc import _copy_data_from_mpc_to_ppc, _adjust_ppc_indices, \
    _change_ppc_TAP_value
from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool


def from_mpc73(mpc_file, f_hz=50, validate_conversion=False, **kwargs):
    mpc = mat73.loadmat(mpc_file)  # , squeeze_me=True, struct_as_record=False)
    ppc = dict()
    ppc['version'] = 'X.X'
    ppc["baseMVA"] = mpc['mpc']['baseMVA']
    ppc["bus"] = mpc['mpc']['bus']
    ppc["gen"] = mpc['mpc']['gen']
    ppc["branch"] = mpc['mpc']['branch']
    ppc['gencost'] = mpc['mpc']['gencost']

    # mpc['mpc']['version'] = 'X.X'
    # mpc['mpc'] = SimpleNamespace(**mpc['mpc'])

    # mpc['mpc'] = np.array(list(mpc['mpc'].items()))#, dtype=dtype)
    # print
    # init empty ppc

    # _copy_data_from_mpc_to_ppc(ppc, mpc, 'mpc')
    _adjust_ppc_indices(ppc)
    _change_ppc_TAP_value(ppc)

    # ppc = _mat2ppc(mpc_file, casename_mpc_file)
    net = pc.from_ppc(ppc, f_hz=f_hz, validate_conversion=validate_conversion, **kwargs)
    return net


class PandaPowerBackend(AbstractBackend):
    """Backend implementation that uses `PandaPower <http://www.pandapower.org>`_."""

    valid_extensions = (".json", ".pkl", ".mat")
    valid_address_names = {
        "bus": ["id"], "load": ["bus", "name"], "sgen": ["bus", "name"], "gen": ["bus", "name"],
        "shunt": ["bus", "name"], "ext_grid": ["bus", "name"], "line": ["from_bus", "to_bus", "name"],
        "trafo": ["hv_bus", "lv_bus", "name"], "poly_cost": ["element"]}
    valid_feature_names = {
        "global": ["converged", "f_hz", "sn_mva"],
        "bus": ["in_service", "max_vm_pu", 'min_vm_pu', "vn_kv", "res_vm_pu", "res_va_degree", "res_p_mw",
            "res_q_mvar"],
        "load": ["const_i_percent", "const_z_percent", "controllable", "in_service", "p_mw", "q_mvar", "scaling",
            "sn_mva", "res_p_mw", "res_q_mvar"],
        "sgen": ["controllable", "in_service", "p_mw", "q_mvar", "scaling", "sn_mva", "current_source", "res_p_mw",
            "res_q_mvar"],
        "gen": ["controllable", "in_service", "p_mw", "scaling", "sn_mva", "vm_pu", "slack", "max_p_mw", "min_p_mw",
            "max_q_mvar", "min_q_mvar", "slack_weight", "res_p_mw", "res_q_mvar", "res_va_degree", "res_vm_pu"],
        "shunt": ["q_mvar", "p_mw", "vn_kv", "step", "max_step", "in_service", "res_p_mw", "res_q_mvar", "res_vm_pu"],
        "ext_grid": ["in_service", "va_degree", "vm_pu", "max_p_mw", "min_p_mw", "max_q_mvar", "min_q_mvar",
            "slack_weight", "res_p_mw", "res_q_mvar"],
        "line": ["c_nf_per_km", "df", "g_us_per_km", "in_service", "length_km", "max_i_ka", "max_loading_percent",
            "parallel", "r_ohm_per_km", "x_ohm_per_km", "res_p_from_mw", "res_q_from_mvar", "res_p_to_mw",
            "res_q_to_mvar", "res_pl_mw", "res_ql_mvar", "res_i_from_ka", "res_i_to_ka", "res_i_ka", "res_vm_from_pu",
            "res_va_from_degree", "res_vm_to_pu", "res_va_to_degree", "res_loading_percent"],
        "trafo": ["df", "i0_percent", "in_service", "max_loading_percent", "parallel", "pfe_kw", "shift_degree",
            "sn_mva", "tap_max", "tap_neutral", "tap_min", "tap_phase_shifter", "tap_pos", "tap_side",
            "tap_step_degree", "tap_step_percent", "vn_hv_kv", "vn_lv_kv", "vk_percent", "vkr_percent", "res_p_hv_mw",
            "res_q_hv_mvar", "res_p_lv_mw", "res_q_lv_mvar", "res_pl_mw", "res_ql_mvar", "res_i_hv_ka", "res_i_lv_ka",
            "res_vm_hv_pu", "res_va_hv_degree", "res_vm_lv_pu", "res_va_lv_degree", "res_loading_percent"],
        "poly_cost": ["cp0_eur", "cp1_eur_per_mw", "cp2_eur_per_mw2", "cq0_eur", "cq1_eur_per_mvar",
            "cq2_eur_per_mvar2"]}

    def __init__(self, n_cores=0):
        """Initializes a PandaPower backend."""
        super().__init__()
        self.n_cores = n_cores
        if self.n_cores > 0:
            self.pool = Pool(self.n_cores)

    def load_network(self, file_path):
        """Loads a pandapower power grid instance, either from a `.pkl` or from a `.json` file.

        Overrides the abstract load_network method.
        """
        if file_path.endswith('.json'):
            net = pp.from_json(file_path)
        elif file_path.endswith('.pkl'):
            net = pp.from_pickle(file_path)
        elif file_path.endswith('.mat'):
            try:
                net = pc.from_mpc(file_path)
            except NotImplementedError:
                net = from_mpc73(file_path)
        else:
            raise NotImplementedError('No support for file {}'.format(file_path))
        net.name = os.path.basename(file_path)
        return net

    def save_network(self, net, path):
        """Saves a power grid instance using the same name as in the initial file.

        Useful for saving a version of a test set modified by a trained neural network.
        Overrides the abstract `save_network` method.
        """
        file_name = net.name
        file_path = os.path.join(path, file_name)
        if file_path.endswith('.json'):
            pp.to_json(net, file_path)
        elif file_path.endswith('.pkl'):
            pp.to_pickle(net, file_path)
        elif file_path.endswith('.mat'):
            pc.to_mpc(net, filename=file_path)
        else:
            raise NotImplementedError('No support for file {}'.format(file_path))

    def set_data_network(self, net, y):
        """Updates a power grid by setting features according to `y`.

        Overrides the abstract `set_feature_network` method.
        """
        for k in y.keys():
            for f in y[k].keys():
                try:
                    net[k][f] = y[k][f]
                except ValueError:
                    print('Object {} and feature {} are not available with PandaPower'.format(k, f))

    def run_network(self, net, **kwargs):
        """Runs a power flow simulation.

        Pandapower `runpp` options can be passed as keyword arguments.
        Overrides the abstract `run_network` method.
        """
        try:
            pp.runpp(net, **kwargs)
        except pp.powerflow.LoadflowNotConverged:
            pass

    def get_data_network(self, network, feature_names=None, address_names=None, address_to_int=True):
        """Extracts features from a pandapower network.

        Addresses are converted into unique integers that start at 0.
        Overrides the abstract `get_data_network` method.
        """
        if feature_names is None:
            feature_names = dict()
        if address_names is None:
            address_names = dict()

        object_names = list(set(list(feature_names.keys()) + list(address_names.keys())))
        x = {}
        for object_name in object_names:

            if (object_name in address_names.keys()) or (object_name in feature_names.keys()):
                x[object_name] = {}
                table = self.get_table(network, object_name)

                object_address_names = address_names.get(object_name, [])
                for address_name in object_address_names:
                    x[object_name][address_name] = table[address_name].astype(str)

                object_feature_names = feature_names.get(object_name, [])
                for feature_name in object_feature_names:
                    x[object_name][feature_name] = np.array(table[feature_name], dtype=np.float32)

        clean_dict(x)
        if address_to_int:
            convert_addresses_to_integers(x, address_names)
        return x

    @staticmethod
    def get_table(net, key):
        """Gets a pandas dataframe describing the features of a specific object in a power grid instance.

        Pandapower puts the results of power flow simulations into a separate table. For instance,
        results at buses is stored in net.res_bus. We thus merge the two table by adding a prefix res
        for the considered features.
        """
        if key == 'bus':
            table = net.bus.copy(deep=True)
            table = table.join(net.res_bus.add_prefix('res_'))
        elif key == 'load':
            table = net.load.copy(deep=True)
            table = table.join(net.res_load.add_prefix('res_'))
            table.name = 'load_' + table.index.astype(str)
        elif key == 'sgen':
            table = net.sgen.copy(deep=True)
            table = table.join(net.res_sgen.add_prefix('res_'))
            table.name = 'sgen_' + table.index.astype(str)
        elif key == 'gen':
            table = net.gen.copy(deep=True)
            table = table.join(net.res_gen.add_prefix('res_'))
            table.name = 'gen_' + table.index.astype(str)
        elif key == 'shunt':
            table = net.shunt.copy(deep=True)
            table = table.join(net.res_shunt.add_prefix('res_'))
            table.name = 'shunt_' + table.index.astype(str)
        elif key == 'ext_grid':
            table = net.ext_grid.copy(deep=True)
            table = table.join(net.res_ext_grid.add_prefix('res_'))
            table.name = 'ext_grid_' + table.index.astype(str)
        elif key == 'line':
            table = net.line.copy(deep=True)
            table = table.join(net.res_line.add_prefix('res_'))
            table.name = 'line_' + table.index.astype(str)
        elif key == 'trafo':
            table = net.trafo.copy(deep=True)
            table = table.join(net.res_trafo.add_prefix('res_'))
            table.name = 'trafo_' + table.index.astype(str)
            table.tap_side = table.tap_side.map({'hv': 0., 'lv': 1.})
        elif key == 'poly_cost':
            table = net.poly_cost.copy(deep=True)
            table['element'] = table.et.astype(str) + '_' + table.element.astype(str)
        elif key == 'global':
            table = pd.DataFrame({'converged': [net.converged], 'f_hz': [net.f_hz], 'sn_mva': [net.sn_mva]})
        else:
            raise ValueError('Object {} is not a valid object name. '.format(key))
        table['id'] = table.index
        table.replace([np.inf], 99999, inplace=True)
        table.replace([-np.inf], -99999, inplace=True)
        table = table.fillna(0.)
        return table

    def run_batch(self, network_batch, **kwargs):
        """Performs power flow computations for a batch of power grids."""

        if self.n_cores > 0:
            n_nets = len(network_batch)
            range_splits = np.array_split(range(n_nets), self.n_cores)

            #def run_single(indices):
            #    [self.run_network(network_batch[i], **kwargs) for i in indices]
            #network_superbatch = np.array_split(network_batch, self.n_cores)
            #self.pool.map(run_single, network_superbatch)
            return self.pool.map(run_single, network_batch)
            #return self.pool.imap(run_single, network_batch)
            #return self.pool.imap_unordered(run_single, network_batch)

            #Parallel(n_jobs=self.n_cores, require='sharedmem')(delayed(run_single)(indices) for indices in range_splits)
        else:
            super(PandaPowerBackend, self).run_batch(network_batch, **kwargs)


def run_single(net):
    pp.runpp(net)
    return net
    #[pp.runpp(net) for net in nets]
