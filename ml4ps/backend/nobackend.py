from ml4ps.backend.interface import AbstractBackend
import os
import json
import numpy as np
from ml4ps.utils import clean_dict, convert_addresses_to_integers


class NoBackend(AbstractBackend):
    """ A way to load a `.json` file without using a backend, it assumes the following structure:
        {'elem1': {'feat1: [...], 'feat2: [...]}, 'elem2': {'feat1: [...], 'feat2: [...]}}

        {'bus':
            {'id': ['A', 'B', 'C'], 'v_min': [0.9, 0.9, 0.9]}
        'load':
            {}
        }

    """

    valid_extensions = (".json")
    valid_address_names = {"branch": ["t_bus", "f_bus"], "load": ["load_bus"], "bus": ["index"], "gen": ["gen_bus"]}
    valid_feature_names = {
        "branch": ['br_r', 'rate_a', 'shift', 'pt', 'mu_sm_fr', 'br_x', 'g_to', 'g_fr', 'b_fr', 'mu_sm_to',
            'br_status', 'b_to', 'index', 'qf', 'angmin', 'angmax', 'transformer', 'qt', 'tap', 'pf'],
        "gen": ['apf', 'qc1max', 'lin_cost', 'quad_cost', 'pg', 'model', 'shutdown', 'startup', 'qc2max',
            'ramp_agc', 'qg', 'pmax',
            'ramp_10', 'vg', 'mbase', 'pc2', 'index', 'cost1', 'cost2', 'qmax', 'gen_status', 'qmin', 'qc1min',
            'qc2min', 'pc1', 'ramp_q', 'ramp_30', 'ncost', 'pmin'],
        "load": ['status', 'qd', 'pd', 'index'],
        "bus": ['zone', 'lam_kcl_r', 'bus_i', 'bus_type', 'vmax', 'col_2', 'col_1', 'area', 'vmin',
            'va', 'lam_kcl_i', 'vm', 'base_kv']}


    def __init__(self, warns=False):
        """Initializes a NoBackend."""
        self.warns = warns
        super().__init__()

    def warning(self):
        if self.warns:
            print("No backend is used")

    def warning_none_return(self):
        self.warning()
        return None

    def load_network(self, file_path):
        """Loads data stored in a `.json` file."""
        with open(file_path, encoding='utf-8') as f:
            net = json.load(f)
        net["name"] = os.path.basename(file_path)
        return net

    def save_network(self, net, path):
        """Saves a power grid instance using the same name as in the initial file.

        Useful for saving a version of a test set modified by a trained neural network.
        Overrides the abstract `save_network` method.
        """
        file_name = net["name"]
        file_path = os.path.join(path, file_name)
        with open(file_path, "w") as outfile:
            outfile.write(net)

    def set_data_network(self, net, y):
        """Updates a power grid by setting features according to `y`."""
        print("Networks cannot be modified.")

    def run_network(self, net, **kwargs):
        """Send a warning"""
        print("Power Flow cannot be run.")

    def get_data_network(self, net, feature_names=None, address_names=None):
        """Returns features from a single power grid instance."""
        if feature_names is None:
            feature_names = dict()
        if address_names is None:
            address_names = dict()

        object_names = list(set(list(feature_names.keys()) + list(address_names.keys())))
        x = {}
        for object_name in object_names:

            if (object_name in address_names.keys()) or (object_name in feature_names.keys()):
                x[object_name] = {}

                object_address_names = address_names.get(object_name, [])
                for address_name in object_address_names:
                    x[object_name][address_name] = np.array(net[object_name][address_name]).astype(str)

                object_feature_names = feature_names.get(object_name, [])
                for feature_name in object_feature_names:
                    if feature_name == 'cost1':
                        x[object_name][feature_name] = np.array(net[object_name]['cost'], dtype=np.float32)[:, 0]
                    elif feature_name == 'cost2':
                        x[object_name][feature_name] = np.array(net[object_name]['cost'], dtype=np.float32)[:, 1]
                    else:
                        x[object_name][feature_name] = np.array(net[object_name][feature_name], dtype=np.float32)

        clean_dict(x)
        convert_addresses_to_integers(x, address_names)
        return x

