import os
import pickle

import pypowsybl.network as pn
from scipy import interpolate
import numpy as np
from tqdm import tqdm

from ML4PS.backend.pypowsybl import PyPowSyblBackend
from ML4PS.backend.pandapower import PandaPowerBackend


class Normalizer:

    def __init__(self, file=None, **kwargs):
        self.features = {}
        self.functions = {}

        if file is not None:
            self.load(file)
        else:
            self.data_dir = kwargs.get("data_dir", None)
            self.backend_name = kwargs.get("backend_name", 'pypowsybl')
            self.shuffle = kwargs.get("shuffle", False)
            self.amount_of_samples = kwargs.get('amount_of_samples', 100)
            self.break_points = kwargs.get('break_points', 200)

            if self.backend_name == 'pypowsybl':
                self.backend = PyPowSyblBackend()
            elif self.backend_name == 'pandapower':
                self.backend = PandaPowerBackend()
            else:
                raise ValueError('Not a valid backend !')

            self.features = kwargs.get("features", self.backend.valid_features)

            self.backend.check_features(self.features)


            self.data_files = self.get_data_files()
            self.build_functions()

    def get_data_files(self):
        all_data_files = []
        train_dir = os.path.join(self.data_dir, 'train')
        for f in sorted(os.listdir(train_dir)):
            if f.endswith(self.backend.valid_extensions):
                all_data_files.append(os.path.join(train_dir, f))

        if not all_data_files:
            raise FileNotFoundError("There is no valid file in {}".format(train_dir))

        if self.shuffle:
            np.random.shuffle(all_data_files)

        return all_data_files[:self.amount_of_samples]

    def build_functions(self):
        dict_of_all_values = self.get_all_values()
        self.functions = {}
        for k in self.features.keys():
            self.functions[k] = {}
            for f in self.features[k]:
                self.functions[k][f] = self.build_single_function(dict_of_all_values[k][f])

    def get_all_values(self):
        values_dict = {k: {f: [] for f in f_list} for k, f_list in self.features.items()}
        for file in tqdm(self.data_files, desc='Loading all the dataset'):
            net = self.backend.load_network(file)
            for k in self.features.keys():
                table = self.backend.get_table(net, k)
                for f in self.features[k]:
                    if (f in table.keys()) and (not table.empty):
                        values_dict[k][f].append(table[f].to_numpy().flatten().astype(float))
        return values_dict

    def build_single_function(self, values):
        if values:
            v, p = self.get_quantiles(values)
            v_unique, p_unique = self.merge_equal_quantiles(v, p)
            if len(v_unique) == 1:
                return SubtractFunction(v_unique[0])
            else:
                return interpolate.interp1d(v_unique, -1 + 2 * p_unique, fill_value="extrapolate")
        else:
            return None

    def get_quantiles(self, values):
        p = np.arange(0, 1, 1. / self.break_points)
        v = np.quantile(values, p)
        return v, p

    def merge_equal_quantiles(self, v, p):
        v_unique, inverse, counts = np.unique(v, return_inverse=True, return_counts=True)
        p_unique = 0. * v_unique
        np.add.at(p_unique, inverse, p)
        p_unique = p_unique / counts
        return v_unique, p_unique

    def save(self, filename):
        file = open(filename, 'wb')
        file.write(pickle.dumps(self.functions))
        file.close()

    def load(self, filename):
        file = open(filename, 'rb')
        self.functions = pickle.loads(file.read())
        file.close()

    def __call__(self, x):
        x_norm = {}
        for k in x.keys():
            if k in self.functions.keys():
                x_norm[k] = {}
                for f in x[k].keys():
                    if (f in self.functions[k].keys()) and (self.functions[k][f] is not None):
                        x_norm[k][f] = self.functions[k][f](x[k][f])
                    else:
                        x_norm[k][f] = x[k][f]
            else:
                x_norm[k] = x[k]
        return x_norm


class SubtractFunction:

    def __init__(self, v):
        self.v = v

    def __call__(self, x):
        return x - self.v
