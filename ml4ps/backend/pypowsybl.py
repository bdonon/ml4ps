from ml4ps.backend.interface import AbstractBackend
from ml4ps.utils import clean_dict, convert_addresses_to_integers

import pypowsybl.loadflow as pl
import pypowsybl.network as pn

import os
import pandas as pd
import numpy as np
from joblib import Parallel, delayed, parallel_backend

VALID_FEATURE_NAMES = {
    'bus': ['v_mag', 'v_angle'],
    'gen': ['target_p', 'min_p', 'max_p', 'min_q', 'max_q', 'target_v', 'target_q', 'voltage_regulator_on',
            'p', 'q', 'i', 'connected'],
    'load': ['p0', 'q0', 'p', 'q', 'i', 'connected'],
    'shunt_compensator': ['g', 'b', 'voltage_regulation_on', 'target_v',  'p','q', 'i', 'connected'],
    "hvdc_line" : ["target_p", "max_p", "nominal_v", "r", "connected1", "connected2"], # hvdc_lines
    #'linear_shunt_compensator_sections': ['g_per_section', 'b_per_section'], # quelle utilité ? 
    'static_var_compensators' : ["b_min", "b_max", "target_v", "target_q"],
    'batteries': ["max_p", "min_p", "min_q", "max_q", "target_p", "target_q", "p", "q","i", "connected"],
    'line': ['r', 'x', 'g1', 'b1', 'g2', 'b2', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'connected1', 'connected2'],
    'twt': ['r', 'x', 'g', 'b', 'rated_u1', 'rated_u2', 'rated_s', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2',
            'connected1', 'connected2'],
    'thwt': ['rated_u0', 'r1', 'x1', 'g1', 'b1', 'rated_u1', 'r2', 'x2', 'g2', 'b2', 'rated_u2', 'r3', 'x3', 'g3', 'b3',
             'rated_u3', 'rated_s1', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'p3', 'q3', 'i3', 'connected1', 'connected2',
             'connected3'],
    'voltage_levels': ['nominal_v', 'high_voltage_limit', 'low_voltage_limit'],
    # current limits => mettre dans "lines" (à réfléchir avec get_operational_limits)
    'operational_limits': ["element_type", "side", "type", "value", "acceptable_duration"],
    # dangling lines
    'dangling_lines': ["r", "x", "g", "b", "p0", "q0", "p", "q", "i", "connected"],
    # ratio_tap_changers => on a le rho pas besoin de recouper 
    'ratio_tap_changers' : ['tap', 'low_tap', 'high_tap', 'on_load', 'regulating', 'target_v', 'target_deadband', 'rho', 'alpha'],
    # phase tap changer recouper avec phase tap changer step (tracer rho en fonction de la position pour chaque tap changer)
    'phase_tap_changers' : ['tap', 'low_tap', 'high_tap', 'regulating', 'regulation_mode', 'regulation_value', 'target_deadband', 'regulated_side', 'rho', 'alpha', 'r', 'x', 'g', 'b'],

    # vsc / lcc converter stations à mettre 
    'vsc_converter_station': ['loss_factor', 'min_q', 'max_q', 'min_q_at_p', 'max_q_at_p', 'reactive_limits_kind', 'target_v', 'target_q', 'voltage_regulator_on', 'p', 'q', 'i' ,'connected'], 
    'lcc_converter_station': ['power_factor', 'loss_factor', 'p', 'q', 'i', 'connected'], 

}

VALID_ADDRESSE_NAMES = {
    'bus': ['id', 'voltage_level_id'],
    'gen': ['id', 'bus_id', 'voltage_level_id'],
    'load': ['id', 'bus_id', 'voltage_level_id'],
    'shunt_compensator': ['id', 'bus_id'],
    #'linear_shunt_compensator_sections': ['id'],
    "static_var_compensators": ['id'],
    'batteries': ["id", "bus_id"],
    'line': ['id', 'bus1_id', 'bus2_id', 'voltage_level1_id', 'voltage_level2_id'],
    'twt': ['id', 'bus1_id', 'bus2_id', 'voltage_level1_id', 'voltage_level2_id'],
    'thwt': ['id', 'bus1_id', 'bus2_id', 'bus3_id'],
    'voltage_levels': ['id', 'name'],
    'substations': ['id', 'name', 'TSO'], # ? 
    'operational_limits': ["id", "element_id"],
    'dangling_lines': ["id", 'bus_id'],
    'ratio_tap_changers' : ['id', 'regulating_bus_id'],
    'phase_tap_changers' : ['id', 'regulating_bus_id'],
    "hvdc_line" : ['id', "converter_station1_id", "converter_station2_id"],
    'vsc_converter_station': ['id', 'bus_id'], 
    'lcc_converter_station': ['id', 'bus_id'], 
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
            table = net.get_buses().reset_index()
        elif key == 'gen':
            table = net.get_generators().reset_index()
        elif key == 'load':
            table = net.get_loads().reset_index()
        elif key == 'shunt_compensator':
            table = net.get_shunt_compensators().reset_index()
        elif key == "static_var_compensators":
            table = net.get_static_var_compensators().reset_index()
        elif key == 'batteries':
            table = net.get_batteries().reset_index()
        elif key == 'line':
            table = net.get_lines().reset_index()
        elif key == 'twt':
            table = net.get_2_windings_transformers().reset_index()
        elif key == 'thwt':
            table = net.get_3_windings_transformers().reset_index()
        elif key == 'voltage_levels':
            table = net.get_voltage_levels().reset_index()
        elif key == 'substations':
            table = net.get_substations().reset_index()
        elif key == 'operational_limits': 
            table = net.get_operational_limits().reset_index()
        elif key == 'dangling_lines': 
            table = net.get_dangling_lines().reset_index()
        elif key == 'ratio_tap_changers': 
            table = net.get_ratio_tap_changers().reset_index()
        elif key == 'phase_tap_changers': 
            table = pd.merge(net.get_phase_tap_changers(all_attributes=True).reset_index(), 
                             net.get_phase_tap_changer_steps().reset_index(), 
                             left_on=['id', 'tap'], right_on=['id', 'position'], how='left')
        elif key == 'hvdc_line': 
            table = net.get_hvdc_lines().reset_index()
        elif key == 'vsc_converter_station': 
            table = net.get_vsc_converter_stations(all_attributes=True).reset_index()
        elif key == 'lcc_converter_station': 
            table = net.get_lcc_converter_stations().reset_index()
        else:
            raise ValueError(f'Object {key} is not a valid object name. ' +
                             f'Please pick from this list : {VALID_FEATURE_NAMES}')
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
                    try:
                        if len(table[feature_name])>0:
                            if isinstance(table[feature_name][0], str):
                                table[feature_name] = table[feature_name].astype('category').cat.codes

                        x[object_name][feature_name] = np.array(table[feature_name], dtype=np.float32)
                    except Exception as e : 
                        print(e, object_name, feature_name)

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
                    net.update_2_windings_transformers(df)
                elif k == 'thwt':
                    df = pd.DataFrame(data=y[k][f], index=net.get_3_windings_transformers().index, columns=[f])
                    net.update_3_windings_transformers(df)
                elif k == 'static_var_compensators': 
                    df = pd.DataFrame(data=y[k][f], index=net.get_static_var_compensators().index, columns=[f])
                    net.update_static_var_compensators(df)
                else:
                    raise ValueError(f'Object {k} is not a valid object name. ' +
                                     f'Please pick from this specific list : {VALID_FEATURE_NAMES}')

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
