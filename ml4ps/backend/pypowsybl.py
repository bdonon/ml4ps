import pypowsybl.loadflow as pl
import pypowsybl.network as pn
import pypowsybl
import os
import pandas as pd
import numpy as np

from ml4ps.backend.interface import AbstractBackend
from ml4ps.h2mg import H2MGStructure, HyperEdgesStructure, H2MG, HyperEdges


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
    """Backend implementation that uses `Pypowsybl <https://pypowsybl.readthedocs.io>`_."""

    valid_extensions = (".xiidm", "xiidm.gz", ".mat")

    valid_structure = H2MGStructure()
    for key in VALID_ADDRESSE_NAMES.keys():
        hyper_edges_structure = HyperEdgesStructure(addresses=VALID_ADDRESSE_NAMES[key], features=VALID_FEATURE_NAMES[key])
        valid_structure.add_local_hyper_edges_structure(key, hyper_edges_structure)

    def __init__(self):
        super().__init__()

    @staticmethod
    def get_table(power_grid: pypowsybl.network, key: str) -> pd.DataFrame:
        """Gets a pandas dataframe for the object class `key`."""
        if key == 'bus':
            table = power_grid.get_buses().reset_index()
        elif key == 'gen':
            table = power_grid.get_generators().reset_index()
        elif key == 'load':
            table = power_grid.get_loads().reset_index()
        elif key == 'shunt_compensator':
            table = power_grid.get_shunt_compensators().reset_index()
        elif key == "static_var_compensators":
            table = power_grid.get_static_var_compensators().reset_index()
        elif key == 'batteries':
            table = power_grid.get_batteries().reset_index()
        elif key == 'line':
            table = power_grid.get_lines().reset_index()
        elif key == 'twt':
            table = power_grid.get_2_windings_transformers().reset_index()
        elif key == 'thwt':
            table = power_grid.get_3_windings_transformers().reset_index()
        elif key == 'voltage_levels':
            table = power_grid.get_voltage_levels().reset_index()
        elif key == 'substations':
            table = power_grid.get_substations().reset_index()
        elif key == 'operational_limits': 
            table = power_grid.get_operational_limits().reset_index()
        elif key == 'dangling_lines': 
            table = power_grid.get_dangling_lines().reset_index()
        elif key == 'ratio_tap_changers': 
            table = power_grid.get_ratio_tap_changers().reset_index()
        elif key == 'phase_tap_changers': 
            table = pd.merge(power_grid.get_phase_tap_changers(all_attributes=True).reset_index(),
                             power_grid.get_phase_tap_changer_steps().reset_index(),
                             left_on=['id', 'tap'], right_on=['id', 'position'], how='left')
        elif key == 'hvdc_line': 
            table = power_grid.get_hvdc_lines().reset_index()
        elif key == 'vsc_converter_station': 
            table = power_grid.get_vsc_converter_stations(all_attributes=True).reset_index()
        elif key == 'lcc_converter_station': 
            table = power_grid.get_lcc_converter_stations().reset_index()
        else:
            raise ValueError(f'Object {key} is not a valid object name. ' +
                             f'Please pick from this list : {VALID_FEATURE_NAMES}')
        table['id'] = table.index
        table.replace([np.inf], 99999, inplace=True)
        table.replace([-np.inf], -99999, inplace=True)
        table = table.fillna(0.)
        return table

    def get_h2mg_from_power_grid(self, power_grid: pypowsybl.network, structure: H2MGStructure = None,
                                 str_to_int: bool = True) -> H2MG:
        """Extracts features from a pypowsybl network.

        Addresses are converted into unique integers that start at 0.
        Overrides the abstract `get_data_network` method.
        """
        if structure is not None:
            current_structure = structure
        elif self.default_structure is not None:
            current_structure = self.default_structure
        else:
            current_structure = self.valid_structure

        h2mg = H2MG()
        if current_structure.local_hyper_edges_structure is not None:
            for k, hyper_edges_structure in current_structure.local_hyper_edges_structure.items():
                table = self.get_table(power_grid, k)
                if not table.empty:

                    addresses_dict = None
                    if hyper_edges_structure.addresses is not None:
                        addresses_dict = {}
                        for address_name in hyper_edges_structure.addresses:
                            addresses_dict[address_name] = table[address_name].astype(str).values

                    features_dict = None
                    if hyper_edges_structure.features is not None:
                        features_dict = {}
                        for feature_name in hyper_edges_structure.features:
                            features_dict[feature_name] = np.nan_to_num(table[feature_name].astype(float).values, copy=False) * 1

                    hyper_edges = HyperEdges(features=features_dict, addresses=addresses_dict)
                    if not hyper_edges.is_empty():
                        h2mg.add_local_hyper_edges(k, hyper_edges)

        if str_to_int:
            h2mg.convert_str_to_int()
        return h2mg

    def load_power_grid(self, file_path: str) -> pypowsybl.network:
        """Loads a pypowsybl power grid instance.

        Overrides the abstract `load_power_grid` method.
        """
        return pn.load(file_path)

    def set_h2mg_into_power_grid(self, power_grid: pypowsybl.network, h2mg: H2MG) -> None:
        """Updates a power grid by setting features according to `h2mg`.

        Overrides the abstract `set_data_power_grid` method.
        """
        for k in h2mg.hyper_edges:
            for f in h2mg[k].features.keys():
                if k == 'bus':
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_buses().index, columns=[f])
                    power_grid.update_buses(df)
                elif k == 'gen':
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_generators().index, columns=[f])
                    power_grid.update_generators(df)
                elif k == 'load':
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_loads().index, columns=[f])
                    power_grid.update_loads(df)
                elif k == 'line':
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_lines().index, columns=[f])
                    power_grid.update_lines(df)
                elif k == 'twt':
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_2_windings_transformers().index, columns=[f])
                    power_grid.update_2_windings_transformers(df)
                elif k == 'thwt':
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_3_windings_transformers().index, columns=[f])
                    power_grid.update_3_windings_transformers(df)
                elif k == 'static_var_compensators': 
                    df = pd.DataFrame(data=h2mg[k].features[f], index=power_grid.get_static_var_compensators().index, columns=[f])
                    power_grid.update_static_var_compensators(df)
                else:
                    raise ValueError(f'Object {k} is not a valid object name. ' +
                                     f'Please pick from this specific list : {VALID_FEATURE_NAMES}')

    def run_power_grid(self, power_grid: pypowsybl.network, **kwargs) -> None:
        """Runs a power flow simulation.

        Pypowsybl `runpp` options can be passed as keyword arguments.
        Overrides the abstract `run_power_grid` method.
        """
        parameters = pl.Parameters(voltage_init_mode=pl.VoltageInitMode.DC_VALUES)
        pl.run_ac(power_grid, parameters=parameters)

    def save_power_grid(self, power_grid: pypowsybl.network, path: str) -> None:
        """Saves a power grid instance using the same name as in the initial file.

        Useful for saving a version of a test set modified by a trained neural network.
        Overrides the abstract `save_network` method.
        """
        file_name = power_grid.name
        file_path = os.path.join(path, file_name)
        if file_path.endswith('.xiidm'):
            power_grid.dump(file_path, format="XIIDM")
        elif file_path.endswith('.mat'):
            power_grid.dump(file_path, format=file_path)
        else:
            raise NotImplementedError('No support for file {}'.format(file_path))
