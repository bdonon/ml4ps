import os
import pickle

import pypowsybl.network as pn
from scipy import interpolate
import numpy as np
from tqdm import tqdm


VALID_FEATURES = {
    'bus': ['v_mag', 'v_angle'],
    'gen': ['target_p', 'min_p', 'max_p', 'min_q', 'max_q', 'target_v', 'target_q', 'voltage_regulator_on',
            'p', 'q', 'i', 'connected'],
    'load': ['p0', 'q0', 'p', 'q', 'i', 'connected'],
    'shunt': ['g', 'b', 'voltage_regulation_on', 'target_v', 'connected'],
    'linear_shunt_compensator_sections': ['g_per_section','b_per_section'],
    'line': ['r', 'x', 'g1', 'b1', 'g2', 'b2', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2', 'connected1', 'connected2'],
    'twt': ['r', 'x', 'g', 'b', 'rated_u1', 'rated_u2', 'rated_s', 'p1', 'q1', 'i1', 'p2', 'q2', 'i2',
            'connected1', 'connected2']
}


class Normalizer:

    # TODO : permettre d'utiliser d'autres backends pour la normalisation des données

    def __init__(self, options=None, file=None):
        self.features = {}
        self.functions = {}
        if options is None and file is None:
            raise AttributeError("Please define options or a file to reload.")
        elif file is not None:
            self.load(file)
        else:
            self.data_dir = options['data_dir']
            self.check_features(options['features'])
            self.get_data_files()
            self.shuffle = options.get('shuffle', False)
            self.amount_of_samples = options.get('amount_of_samples', 100)
            self.break_points = options.get('break_points', 10)
            if self.shuffle:
                np.random.shuffle(self.data_files)
            self.build_functions()

    def get_data_files(self):
        self.data_files = []
        # Get list of valid files
        for f in sorted(os.listdir(self.data_dir)):
            if f.endswith((".mat", ".xiidm")):
                self.data_files.append(os.path.join(self.data_dir, f))
        # Make sure that there is at least one valid file
        if not self.data_files:
            raise FileNotFoundError("There is no valid file in {}".format(self.data_dir))

    def check_features(self, features):
        for k in features.keys():
            if k in VALID_FEATURES.keys():
                self.features[k] = []
                for f in features[k]:
                    if f in VALID_FEATURES[k]:
                        self.features[k].append(f)
                    else:
                        raise Warning('{} is not a valid feature for {}. '.format(f,k) + \
                                      'Please pick from this list : {}'.format(VALID_FEATURES[k]))
            else:
                raise Warning('{} is not a valid name. Please pick from this list : {}'.format(k, VALID_FEATURES))

    def build_functions(self):
        dict_of_all_values = self.get_all_values()
        self.functions = {}
        for k in self.features.keys():
            self.functions[k] = {}
            for f in self.features[k]:
                self.functions[k][f] = self.build_single_function(dict_of_all_values[k][f])

    def get_all_values(self):
        values_dict = {}
        for k in self.features.keys():
            values_dict[k] = {}
            for f in self.features[k]:
                values_dict[k][f] = []

        for file in tqdm(self.data_files[:self.amount_of_samples], desc='Loading all the dataset'):
            net = pn.load(file)
            for k in self.features.keys():
                if k == 'bus':
                    table = net.get_buses()
                elif k == 'gen':
                    table = net.get_generators()
                elif k == 'load':
                    table = net.get_loads()
                elif k == 'shunt':
                    table = net.get_shunt_compensators()
                elif k == 'line':
                    table = net.get_lines()
                elif k == 'twt':
                    table = net.get_2_windings_transformers()
                elif k == 'linear_shunt_compensator_sections':
                    table = net.get_linear_shunt_compensator_sections()
                else:
                    raise ValueError('Object {} is not a valid object name. ' + \
                                     'Please pick from this list : {}'.format(k, VALID_FEATURES))
                for f in self.features[k]:
                    # TODO : remplacer inf par autre chose que 0
                    table[f].replace([np.inf, -np.inf], 0, inplace=True)
                    values_dict[k][f].append(table[f].fillna(0.).to_numpy().flatten()*1.)
        return values_dict

    def build_single_function(self, values):
        # Get pairs of quantiles and percentages
        v, p = self.get_quantiles(values)
        # Get rid of identical values for quantiles by merging them together
        v_unique, p_unique = self.merge_equal_quantiles(v, p)
        if len(v_unique) == 1:
            subtract_function = SubtractFunction(v_unique[0])
            return subtract_function
        else:
            return interpolate.interp1d(v_unique, -1 + 2 * p_unique, fill_value="extrapolate")

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
                    if f in self.functions[k].keys():
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
