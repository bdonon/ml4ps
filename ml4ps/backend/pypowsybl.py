from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, convert_addresses_to_integers

import pypowsybl.loadflow as pl
import pypowsybl.network as pn

import os
import pandas as pd
import numpy as np

VALID_FEATURE_NAMES = {
    'bus': ['v_mag', 'v_angle'],
    'gen': ['target_p', 'min_p', 'max_p', 'min_q', 'max_q', 'target_v', 'target_q', 'voltage_regulator_on',
            'p', 'q', 'i', 'connected'],
    'load': ['p0', 'q0', 'p', 'q', 'i', 'connected'],
    'shunt': ['g', 'b', 'voltage_regulation_on', 'target_v', 'connected'],
    'linear_shunt_compensator_sections': ['g_per_section', 'b_per_section'],
    'static_var_compensators' : ["b_min", "b_max", "target_v", "target_q"],
    'batteries': ["connected"],
    'line': ['r', 'x', 'g1', 'b1', 'g2', 'b2', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'connected1', 'connected2'],
    'twt': ['r', 'x', 'g', 'b', 'rated_u1', 'rated_u2', 'rated_s', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2',
            'connected1', 'connected2'],
    'thwt': ['rated_u0', 'r1', 'x1', 'g1', 'b1', 'rated_u1', 'r2', 'x2', 'g2', 'b2', 'rated_u2', 'r3', 'x3', 'g3', 'b3',
             'rated_u3', 'rated_s1', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'p3', 'q3', 'i3', 'connected1', 'connected2',
             'connected3']
}
VALID_ADDRESSE_NAMES = {
    'bus': ['id'],
    'gen': ['id', 'bus_id'],
    'load': ['id', 'bus_id'],
    'shunt': ['id', 'bus_id'],
    'linear_shunt_compensator_sections': ['id'],
    'batteries': ["id", "bus_id"],
    'line': ['id', 'bus1_id', 'bus2_id'],
    'twt': ['id', 'bus1_id', 'bus2_id'],
    'thwt': ['id', 'bus1_id', 'bus2_id', 'bus3_id'],
}


class PyPowSyblBackend(AbstractBackend):

    valid_extensions = (".xiidm", "xiidm.gz", ".mat")
    valid_feature_names = VALID_FEATURE_NAMES
    valid_address_names = VALID_ADDRESSE_NAMES

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_table(net, key):
        if key == 'bus':
            table = net.get_buses()
        elif key == 'gen':
            table = net.get_generators()
        elif key == 'load':
            table = net.get_loads()
        elif key == 'shunt':
            table = net.get_shunt_compensators()
        elif key == 'batteries':
            table = net.get_batteries()
        elif key == 'line':
            table = net.get_lines()
        elif key == 'twt':
            table = net.get_2_windings_transformers()
        elif key == "thwt":
            table = net.get_3_windings_transformers()
        elif key == 'linear_shunt_compensator_sections':
            table = net.get_linear_shunt_compensator_sections()
        elif key == "static_var_compensators":
            table = net.get_static_var_compensators()
        else:
            raise ValueError('Object {} is not a valid object name. ' +
                             'Please pick from this list : {}'.format(key, VALID_FEATURE_NAMES))
        table['id'] = table.index
        table.replace([np.inf], 99999, inplace=True)
        table.replace([-np.inf], -99999, inplace=True)
        table = table.fillna(0.)
        return table

    def get_data_network(self, network, feature_names=None, address_names=None):
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
        convert_addresses_to_integers(x, address_names)
        return x

    def load_network(self, file_path):
        return pn.load(file_path)

    def set_data_network(self, net, y):
        for k in y.keys():
            for f in y[k].keys():
                if k == 'bus':
                    df = pd.DataFrame(data=y[k][f], index=net.get_buses().index, columns=[f])
                    net.update_buses(df)
                elif k == 'gen':
                    df = pd.DataFrame(data=y[k][f], index=net.get_generators().index, columns=[f])
                    net.update_generators(df)
                elif k == 'load':
                    df = pd.DataFrame(data=y[k][f], index=net.get_loads().index, columns=[f])
                    net.update_loads(df)
                elif k == 'line':
                    df = pd.DataFrame(data=y[k][f], index=net.get_lines().index, columns=[f])
                    net.update_loads(df)
                elif k == 'twt':
                    df = pd.DataFrame(data=y[k][f], index=net.get_2_windings_transformers().index, columns=[f])
                    net.update_loads(df)
                else:
                    raise ValueError('Object {} is not a valid object name. ' +
                                     'Please pick from this list : {}'.format(k, VALID_FEATURE_NAMES))

    def run_network(self, net, **kwargs):
        pl.run_ac(net, **kwargs)

    def save_network(self, net, path):
        """Saves a power grid instance using the same name as in the initial file.

        Useful for saving a version of a test set modified by a trained neural network.
        Overrides the abstract `save_network` method.
        """
        file_name = net.name
        file_path = os.path.join(path, file_name)
        if file_path.endswith('.xiidm'):
            net.dump(file_path, format="XIIDM")
        elif file_path.endswith('.mat'):
            net.dump(file_path, format=file_path)
        else:
            raise NotImplementedError('No support for file {}'.format(file_path))
