from ML4PS.backend.abstractbackend import AbstractBackend
import pypowsybl.loadflow as pl
import pypowsybl.network as pn
import pandas as pd
import numpy as np

VALID_FEATURES = {
    'bus': ['v_mag', 'v_angle'],
    'gen': ['target_p', 'min_p', 'max_p', 'min_q', 'max_q', 'target_v', 'target_q', 'voltage_regulator_on',
            'p', 'q', 'i', 'connected'],
    'load': ['p0', 'q0', 'p', 'q', 'i', 'connected'],
    'shunt': ['g', 'b', 'voltage_regulation_on', 'target_v', 'connected'],
    'linear_shunt_compensator_sections': ['g_per_section', 'b_per_section'],
    'line': ['r', 'x', 'g1', 'b1', 'g2', 'b2', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'connected1', 'connected2'],
    'twt': ['r', 'x', 'g', 'b', 'rated_u1', 'rated_u2', 'rated_s', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2',
            'connected1', 'connected2']
}
VALID_ADDRESSES = {
    'bus': ['id'],
    'gen': ['id', 'bus_id'],
    'load': ['id', 'bus_id'],
    'shunt': ['id', 'bus_id'],
    'linear_shunt_compensator_sections': ['id'],
    'line': ['id', 'bus1_id', 'bus2_id'],
    'twt': ['id', 'bus1_id', 'bus2_id']
}

class PyPowSyblBackend(AbstractBackend):

    valid_extensions = (".xiidm", ".mat")
    valid_features = VALID_FEATURES
    valid_addresses = VALID_ADDRESSES

    def __init__(self):
        super().__init__()

    def get_table(self, net, key):
        if key == 'bus':
            table = net.get_buses()
        elif key == 'gen':
            table = net.get_generators()
        elif key == 'load':
            table = net.get_loads()
        elif key == 'shunt':
            table = net.get_shunt_compensators()
        elif key == 'line':
            table = net.get_lines()
        elif key == 'twt':
            table = net.get_2_windings_transformers()
        elif key == 'linear_shunt_compensator_sections':
            table = net.get_linear_shunt_compensator_sections()
        else:
            raise ValueError('Object {} is not a valid object name. ' +
                             'Please pick from this list : {}'.format(key, VALID_FEATURES))
        table['id'] = table.index
        table.replace([np.inf], 99999, inplace=True)
        table.replace([-np.inf], -99999, inplace=True)
        table = table.fillna(0.)
        return table

    def load_network(self, file_path):
        return pn.load(file_path)

    def update_network(self, net, y):
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
                    df = pd.DataFrame(data=y[k][f], index=net.get_loads().index, columns=[f])
                    net.update_loads(df)
                elif k == 'twt':
                    df = pd.DataFrame(data=y[k][f], index=net.get_loads().index, columns=[f])
                    net.update_loads(df)
                else:
                    raise ValueError('Object {} is not a valid object name. ' +
                                     'Please pick from this list : {}'.format(k, VALID_FEATURES))

    def run_load_flow(self, net, load_flow_options=None):
        # TODO : connect options
        pl.run_ac(net)

