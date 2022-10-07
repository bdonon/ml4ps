from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, convert_addresses_to_integers#, build_unique_id_dict
import pandapower as pp
import numpy as np
import json
import sys
import os


class PandaPowerBackend(AbstractBackend):
    """Backend implementation that uses `PandaPower <http://www.pandapower.org>`_."""

    valid_extensions = (".json", ".pkl")
    with open(os.path.join(os.path.dirname(__file__), 'pandapower_data_structure.json'), 'r') as f:
        valid_data_structure = json.load(f)

    # valid_feature_names = VALID_FEATURE_NAMES
    # valid_address_names = VALID_ADDRESS_NAMES

    def __init__(self):
        """Initializes a PandaPowerBackend."""
        super().__init__()

    def load_network(self, file_path):
        """Loads a power grid instance, either from a `.pkl` or from a `.json` file."""
        if file_path.endswith('.json'):
            net = pp.from_json(file_path)
        elif file_path.endswith('.pkl'):
            net = pp.from_pickle(file_path)
        else:
            raise NotImplementedError('No support for file {}'.format(file_path))
        return net

    def set_feature_network(self, net, y):
        """Updates a power grid by setting features according to `y`."""
        for k in y.keys():
            for f in y[k].keys():
                try:
                    net[k][f] = y[k][f]
                except ValueError:
                    print('Object {} and key {} are not available with PandaPower'.format(k, f))

    def run_network(self, net, **kwargs):
        """Runs a power flow simulation."""
        try:
            pp.runpp(net, **kwargs)
        except pp.powerflow.LoadflowNotConverged:
            pass

    def get_data_network(self, network, data_structure):
        """"""
        x = {}
        for k in data_structure.keys():
            if k == 'global':
                x[k] = get_global_features(network, data_structure[k])
            else:
                x[k] = get_local_features(network, data_structure[k], k)
        clean_dict(x)
        convert_addresses_to_integers(x)
        return x


def get_global_features(network, feature_names):
    """"""
    r = {}
    for name in feature_names:
        if name == 'converged':
            r[name] = np.array([network.converged], dtype=np.float32)
        elif name == 'f_hz':
            r[name] = np.array([network.f_hz], dtype=np.float32)
        elif name == 'sn_mva':
            r[name] = np.array([network.sn_mva], dtype=np.float32)
        else:
            raise ValueError('{} not an available global feature.'.format(name))
    return r


def get_local_features(network, structure, object_name):
    """"""
    table = get_table(network, object_name)
    r = {}
    address_names = structure.get('address_names', None)
    if address_names is not None:
        r["address"] = {a: table[a].astype(str) for a in address_names}
    feature_names = structure.get('feature_names', None)
    if feature_names is not None:
        r["features"] = {a: np.array(table[a], dtype=np.float32) for a in feature_names}
    return r




        # table_dict = self.get_table_dict(network, k)
        # address_names = data_structure[k]['address_names']
        # if address_names:
        #    x["address"] = {}
        #    for f in address_names:
        #        x[["address"] = table_dict[f]

    # def get_feature_network(self, network, feature_names):
    #     """Returns features from a single power grid instance."""
    #     table_dict = self.get_table_dict(network, feature_names)
    #     x = {k: {f: np.array(xkf, dtype=np.float32) for f, xkf in xk.items()} for k, xk in table_dict.items()}
    #     return clean_dict(x)
    #
    # def get_address_network(self, network, address_names):
    #     """Extracts a nested dict of address ids from a power grid instance."""
    #     table_dict = self.get_table_dict(network, address_names)
    #     id_dict = build_unique_id_dict(table_dict, address_names)
    #     a = {k: {f: np.array(xkf.astype(str).map(id_dict), dtype=np.int32) for f, xkf in xk.items()}
    #          for k, xk in table_dict.items()}
    #     return clean_dict(a)


# def get_table_dict(network, feature_names):
#     """Gets a dict of pandas tables for all the objects in feature_names, from the input network."""
#     return {k: get_table(network, k, f) for k, f in feature_names.items()}


def get_table(net, key):  # , feature_list):
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
    else:
        raise ValueError('Object {} is not a valid object name. ' +
                         'Please pick from : {}'.format(key, self.valid_feature_names))
    table['id'] = table.index
    table.replace([np.inf], 99999, inplace=True)
    table.replace([-np.inf], -99999, inplace=True)
    table = table.fillna(0.)
    # features_to_keep = list(set(list(table)) & set(feature_list))
    # return table[features_to_keep]
    return table
