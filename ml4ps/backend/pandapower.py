import pandapower.converter as pc
import pandapower as pp
import pandas as pd
import numpy as np
import mat73
import os

from pandapower.converter.matpower.from_mpc import _adjust_ppc_indices, _change_ppc_TAP_value
from multiprocessing import Pool

from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, convert_addresses_to_integers, collate_dict, separate_dict


def from_mpc73(mpc_file, f_hz=50, validate_conversion=False, **kwargs):
    """Implementation of from_mpc that supports matlab 7.3 .m format."""

    mpc = mat73.loadmat(mpc_file)
    ppc = dict()
    ppc['version'] = 'X.X'
    ppc["baseMVA"] = mpc['mpc']['baseMVA']
    ppc["bus"] = mpc['mpc']['bus']
    ppc["gen"] = mpc['mpc']['gen']
    ppc["branch"] = mpc['mpc']['branch']
    ppc['gencost'] = mpc['mpc']['gencost']

    _adjust_ppc_indices(ppc)
    _change_ppc_TAP_value(ppc)

    return pc.from_ppc(ppc, f_hz=f_hz, validate_conversion=validate_conversion, **kwargs)


class PandaPowerBackend(AbstractBackend):
    """Backend implementation that uses `PandaPower <http://www.pandapower.org>`_."""

    valid_extensions = (".json", ".pkl", ".mat")
    valid_address_names = {
        "bus": ["id", "name"], "load": ["id", "name", "bus_id"], "sgen": ["id", "name", "bus_id"],
        "gen": ["id", "name", "bus_id"], "shunt": ["id", "name", "bus_id"], "ext_grid": ["id", "name", "bus_id"],
        "line": ["id", "name", "from_bus_id", "to_bus_id"], "trafo": ["id", "name", "hv_bus_id", "lv_bus_id"],
        "poly_cost": ["element"]}
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

    @staticmethod
    def set_data_network(net, y):
        """Updates a power grid by setting features according to `y`.

        Overrides the abstract `set_feature_network` method.
        """
        for k in y.keys():
            for f in y[k].keys():
                try:
                    net[k][f] = y[k][f]
                except ValueError:
                    print('Object {} and feature {} are not available with PandaPower'.format(k, f))

    # def set_data_network(self, net, y):
    #     """Updates a power grid by setting features according to `y`.
    #
    #     Overrides the abstract `set_feature_network` method.
    #     """
    #     for k in y.keys():
    #         for f in y[k].keys():
    #             try:
    #                 net[k][f] = y[k][f]
    #             except ValueError:
    #                 print('Object {} and feature {} are not available with PandaPower'.format(k, f))

    @staticmethod
    def run_network(net, **kwargs):
        """Runs a power flow simulation.

        Pandapower `runpp` options can be passed as keyword arguments.
        Overrides the abstract `run_network` method.
        """
        try:
            pp.runpp(net, **kwargs)
        except pp.powerflow.LoadflowNotConverged:
            pass

    @staticmethod
    def get_data_network(network, feature_names=None, address_names=None, address_to_int=True):
        """Extracts features from a pandapower network.

        If `address_to_int` is True, addresses are converted into unique integers that start at 0.
        Overrides the abstract `get_data_network` method.
        """
        if feature_names is None:
            feature_names = dict()
        if address_names is None:
            address_names = dict()

        object_names = list(set(list(feature_names.keys()) + list(address_names.keys())))
        x = {}
        for key in object_names:
            if key == 'global':
                x[key] = pd.DataFrame({
                    'converged': [network.converged * 1.], 'f_hz': [network.f_hz * 1.], 'sn_mva': [network.sn_mva * 1.]})
            else:
                x[key] = {}
                table = network.get(key)
                res_table = network.get('res_' + key)
                for feature_name in feature_names.get(key, []):
                    if feature_name == 'tap_side':
                        x[key][feature_name] = table[feature_name].map({'hv': 0., 'lv': 1.}).astype(float).values
                    elif feature_name[:4] == 'res_':
                        x[key][feature_name] = res_table[feature_name[4:]].astype(float).values
                    else:
                        x[key][feature_name] = table[feature_name].astype(float).values
                    x[key][feature_name] = np.nan_to_num(x[key][feature_name], copy=False) * 1

                for address_name in address_names.get(key, []):
                    if address_name == 'id':
                        x[key][address_name] = (key + '_' + table.index.astype(str)).values
                    elif address_name == 'bus_id':
                        x[key][address_name] = ('bus_' + table.bus.astype(str)).values
                    elif address_name == 'from_bus_id':
                        x[key][address_name] = ('bus_' + table.from_bus.astype(str)).values
                    elif address_name == 'to_bus_id':
                        x[key][address_name] = ('bus_' + table.to_bus.astype(str)).values
                    elif address_name == 'hv_bus_id':
                        x[key][address_name] = ('bus_' + table.hv_bus.astype(str)).values
                    elif address_name == 'lv_bus_id':
                        x[key][address_name] = ('bus_' + table.lv_bus.astype(str)).values
                    elif address_name == 'element':
                        x[key][address_name] = table.et.astype(str) + '_' + table.element.astype(str)
                    else:
                        x[key][address_name] = table[address_name].values.astype(str)
        clean_dict(x)
        if address_to_int:
            convert_addresses_to_integers(x, address_names)
        return x

    def set_run_get_batch(self, network_batch, mod_dict, feature_names, **kwargs):
        """Applies a list of modifications to a batch of power grids, runs simulations, and returns features.

        Args:
            network_batch (:obj:`list` of :obj:`pandapower.Network`): Batch of `N_batch` pandapower networks.
            mod_dict (:obj:`dict` of :obj:`dict` of :obj:`Array`): Dictionary of dictionary of 3D tensors.
                The upper level keys corresponds to object classes and the lower level keys to feature names.
                Tensors are of dimension `[N_batch, N_variant, N_object]`, where `N_batch` is the amount of
                samples per batch, `N_variant` is the amount of modifications one wants to apply to each
                sample, and `N_object` is the amount of objects of the considered class.
            feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary of list of object features that
                should be extracted from the results of the power flow computations.
            **kwargs: Any keyword argument that should be passed to `pandapower.runpp`.
        """

        mod_batch_variant = [separate_dict(m) for m in separate_dict(mod_dict)]

        if self.n_cores > 0:
            # Multiprocessing implementation
            args = [(network, mod_variant, feature_names, kwargs) for network, mod_variant in
                zip(network_batch, mod_batch_variant)]
            out_list = self.pool.map(PandaPowerBackend.set_run_get_single, args)
            return collate_dict(out_list)

        else:
            # Standard implementation
            features_batch_variant = []
            for network, mod_variant in zip(network_batch, mod_batch_variant):
                features_variant = []
                for mod in mod_variant:
                    PandaPowerBackend.set_data_network(network, mod)
                    PandaPowerBackend.run_network(network, **kwargs)
                    features = PandaPowerBackend.get_data_network(network, feature_names=feature_names)
                    features_variant.append(features)
                features_batch_variant.append(collate_dict(features_variant))
            return collate_dict(features_batch_variant)

    @staticmethod
    def set_run_get_single(args):
        """Applies a list of modifications to a single net, runs power flows, and returns required features."""
        network, mod_variant, feature_names, kwargs = args
        features_variant = []
        for mod in mod_variant:
            PandaPowerBackend.set_data_network(network, mod)
            PandaPowerBackend.run_network(network, **kwargs)
            features = PandaPowerBackend.get_data_network(network, feature_names=feature_names)
            features_variant.append(features)
        return collate_dict(features_variant)
