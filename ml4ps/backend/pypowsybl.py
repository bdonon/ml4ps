from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, convert_addresses_to_integers

import pypowsybl.loadflow as pl
import pypowsybl.network as pn

import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed,parallel_backend

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
             'connected3'],
    'voltage_levels': ['nominal_v', 'high_voltage_limit', 'low_voltage_limit']
}
VALID_ADDRESSE_NAMES = {
    'bus': ['id', 'voltage_level_id'],
    'gen': ['id', 'bus_id', 'voltage_level_id'],
    'load': ['id', 'bus_id', 'voltage_level_id'],
    'shunt': ['id', 'bus_id'],
    'linear_shunt_compensator_sections': ['id'],
    "static_var_compensators": ['id'],
    'batteries': ["id", "bus_id"],
    'line': ['id', 'bus1_id', 'bus2_id', 'voltage_level1_id', 'voltage_level2_id'],
    'twt': ['id', 'bus1_id', 'bus2_id', 'voltage_level1_id', 'voltage_level2_id'],
    'thwt': ['id', 'bus1_id', 'bus2_id', 'bus3_id'],
    'voltage_levels': ['id', 'name'],
    'substations': ['id', 'name', 'TSO']
}


class PyPowSyblBackend(AbstractBackend):

    valid_extensions = (".xiidm", "xiidm.gz", ".mat")
    valid_feature_names = VALID_FEATURE_NAMES
    valid_address_names = VALID_ADDRESSE_NAMES

    def __init__(self, n_cores=0):
        super().__init__()

        self.n_cores = n_cores

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
        elif key == 'thwt':
            table = net.get_3_windings_transformers()
        elif key == 'linear_shunt_compensator_sections':
            table = net.get_linear_shunt_compensator_sections()
        elif key == "static_var_compensators":
            table = net.get_static_var_compensators()
        elif key == 'voltage_levels':
            table = net.get_voltage_levels()
        elif key == 'substations':
            table = net.get_substations()
        else:
            raise ValueError('Object {} is not a valid object name. ' +
                             'Please pick from this list : {}'.format(key, VALID_FEATURE_NAMES))
        table['id'] = table.index
        table.replace([np.inf], 99999, inplace=True)
        table.replace([-np.inf], -99999, inplace=True)
        table = table.fillna(0.)
        return table

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
                    x[object_name][address_name] = table[address_name].astype(str).values

                object_feature_names = feature_names.get(object_name, [])
                for feature_name in object_feature_names:
                    x[object_name][feature_name] = np.array(table[feature_name], dtype=np.float32)

        clean_dict(x)
        if address_to_int:
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
                    net.update_lines(df)
                elif k == 'twt':
                    df = pd.DataFrame(data=y[k][f], index=net.get_2_windings_transformers().index, columns=[f])
                    net.update(df)
                else:
                    raise ValueError('Object {} is not a valid object name. ' +
                                     'Please pick from this specific list : {}'.format(k, VALID_FEATURE_NAMES))

    def run_network(self, net, **kwargs):
        parameters = pl.Parameters(voltage_init_mode=pl.VoltageInitMode.DC_VALUES)
        pl.run_ac(net, parameters=parameters)

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

    def run_batch(self, network_batch, **kwargs):
        """Performs power flow computations for a batch of power grids."""
        def run_single(i):
            net = network_batch[i]
            parameters = pl.Parameters(voltage_init_mode=pl.VoltageInitMode.DC_VALUES)
            pl.run_ac(net, parameters=parameters)

        if self.n_cores > 0:
            Parallel(n_jobs=self.n_cores, require='sharedmem')(delayed(run_single)(i) for i in range(len(network_batch)))
        else:
            super(PyPowSyblBackend, self).run_batch(network_batch, **kwargs)

