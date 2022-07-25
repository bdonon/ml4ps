import pypowsybl.loadflow as pl
import pypowsybl.network as pn
import pandas as pd
import numpy as np
import math
import os


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


class Iterator:

    def __init__(self, files, shuffle, interface):

        self.batch_files = None
        self.length = None
        self.files = files
        self.shuffle = shuffle
        self.batch_size = interface.batch_size
        self.series_length = interface.series_length
        self.time_window = interface.time_window
        self.addresses = interface.addresses
        self.features = interface.features
        self.keys = list(set(list(self.features.keys()) + list(self.addresses.keys())))
        self.initialize_batch_order()

    def load_sample(self, file):
        net = pn.load(file)
        x = {}
        a = {}
        all_addresses = []
        for k in self.keys:
            if k in self.features.keys():
                x[k] = {}
            if k in self.addresses.keys():
                a[k] = {}
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
                raise ValueError('Object {} is not a valid object name. ' +
                                 'Please pick from this list : {}'.format(k, VALID_FEATURES))

            if k in self.features.keys():
                for f in self.features[k]:
                    # TODO : remplacer inf par autre chose que 0
                    table[f].replace([np.inf, -np.inf], 0, inplace=True)
                    x[k][f] = table[f].fillna(0.).to_numpy().flatten() * 1.
            if k in self.addresses.keys():
                for f in self.addresses[k]:
                    if f == 'id':
                        address_list = list(table.index)
                    else:
                        address_list = list(table[f])
                    for address in address_list:
                        if address not in all_addresses:
                            all_addresses.append(address)
                    a[k][f] = np.array([all_addresses.index(i) for i in address_list])
        return a, x, net

    def concatenate(self, a_batch, x_batch):
        # TODO : One should be able to concatenate networks of varying size
        x_concat = {}
        a_concat = {}
        for k in self.keys:
            if k in self.features.keys():
                x_concat[k] = {}
                for f in self.features[k]:
                    x_concat[k][f] = []
                    for x_window in x_batch:
                        temp = []
                        for x_sample in x_window:
                            temp.append(x_sample[k][f])
                        temp = np.array(temp)
                        x_concat[k][f].append(temp)
                    x_concat[k][f] = np.array(x_concat[k][f])
                    x_concat[k][f] = np.transpose(x_concat[k][f], [0,2,1])
            if k in self.addresses.keys():
                a_concat[k] = {}
                for f in self.addresses[k]:
                    a_concat[k][f] = []
                    for a_window in a_batch:
                        # Check that topology remains constant ?
                        for a_sample in a_window:
                            assert np.array_equal(a_sample[k][f], a_window[0][k][f])
                        temp = np.array([a_window[0][k][f]])
                        a_concat[k][f].append(temp)

                    temp = np.array(temp)
                    a_concat[k][f] = np.array(a_concat[k][f])
                    a_concat[k][f] = np.transpose(a_concat[k][f], [0,2,1])

        if False:
            n_obj = 0
            for k in self.addresses.keys():
                for f in self.addresses[k]:
                    n_obj = np.maximum(n_obj, np.max(a_concat[k][f]) + 1)
            offset = n_obj * np.reshape(np.arange(self.batch_size), [-1, 1])
            for k in self.addresses.keys():
                for f in self.addresses[k]:
                    a_concat[k][f] = np.array(a_concat[k][f])
                    a_concat[k][f] += offset

        return a_concat, x_concat

    def initialize_batch_order(self):
        n_files = len(self.files)
        sl = self.series_length
        tw = self.time_window
        window_files = []
        for i in range(0, n_files, sl):
            for j in range(0, sl - tw + 1):
                subseries = self.files[i + j:i + j + tw]
                if len(subseries) == tw:
                    window_files.append(subseries)
                else:
                    continue

        # Shuffle samples
        if self.shuffle:
            np.random.shuffle(window_files)

        # Split in batches
        bs = self.batch_size
        nb = len(window_files)
        self.batch_files = [window_files[i:i+bs] for i in range(0, nb, bs)]
        self.length = len(self.batch_files)

    def __iter__(self):
        self.current_batch_id = 0
        self.initialize_batch_order()
        return self

    def __len__(self):
        return self.length

    def __next__(self, mode='train'):
        if self.current_batch_id < len(self.batch_files):
            current_batch_files = self.batch_files[self.current_batch_id]
            a_batch = []
            x_batch = []
            net_batch = []
            for window in current_batch_files:
                a_window = []
                x_window = []
                net_window = []
                for file in window:
                    a_sample, x_sample, net_sample = self.load_sample(file)
                    a_window.append(a_sample)
                    x_window.append(x_sample)
                    net_window.append(net_sample)
                a_batch.append(a_window)
                x_batch.append(x_window)
                net_batch.append(net_window)
            a_batch, x_batch = self.concatenate(a_batch, x_batch)
            self.current_batch_id += 1
        else:
            raise StopIteration
        return a_batch, x_batch, net_batch


class Interface:

    def __init__(self, options):

        self.batch_files = None
        self.addresses = {}
        self.features = {}

        self.validation_portion = options.get('validation_portion', [0.9, 1.0])
        self.batch_size = options.get('batch_size', 10)
        self.shuffle = options.get('shuffle', False)
        self.series_length = options.get('series_length', 1)
        self.time_window = options.get('time_window', 1)

        self.check_data_dir(options['data_dir'])
        self.check_features(options['features'])
        self.check_addresses(options['addresses'])
        self.keys = list(set(list(self.features.keys()) + list(self.addresses.keys())))

        self.train = Iterator(self.train_files, self.shuffle, self)
        self.val = Iterator(self.val_files, False, self)
        self.test = Iterator(self.test_files, False, self)

        self.backend = 'PYPOWSYBL' # TODO : implem d'autres backend, comme pandapower

    def check_data_dir(self, data_dir):
        self.data_dir = data_dir

        self.train_dir = os.path.join(self.data_dir, 'train')
        self.train_files = []
        for f in sorted(os.listdir(self.train_dir)):
            if f.endswith((".mat", ".xiidm")):
                self.train_files.append(os.path.join(self.train_dir, f))
            else:
                continue

        # Build validation set, while respecting time windows
        n_train = len(self.train_files)
        val_start_id = math.floor(self.validation_portion[0] * n_train / self.series_length) * self.series_length
        val_end_id = math.ceil(self.validation_portion[1] * n_train / self.series_length) * self.series_length
        self.val_files = self.train_files[val_start_id:val_end_id]
        self.train_files = self.train_files[:val_start_id] + self.train_files[val_end_id:]

        self.test_dir = os.path.join(self.data_dir, 'test')
        self.test_files = []
        for f in sorted(os.listdir(self.test_dir)):
            if f.endswith((".mat", ".xiidm")):
                self.test_files.append(os.path.join(self.test_dir, f))
            else:
                continue

        # Make sure that there is at least one valid file in all sets
        if (not self.train_files) and (not self.val_files) and (not self.test_files):
            raise FileNotFoundError("There is no valid file in {}".format(self.data_dir))

    def check_features(self, features):
        for k in features.keys():
            if k in VALID_FEATURES.keys():
                self.features[k] = []
                for f in features[k]:
                    if f in VALID_FEATURES[k]:
                        self.features[k].append(f)
                    else:
                        raise Warning('{} is not a valid feature for {}. '.format(f, k) +
                                      'Please pick from this list : {}'.format(VALID_FEATURES[k]))
            else:
                raise Warning('{} is not a valid name. Please pick from this list : {}'.format(k, VALID_FEATURES))

    def check_addresses(self, addresses):
        for k in addresses.keys():
            if k in VALID_ADDRESSES.keys():
                self.addresses[k] = []
                for f in addresses[k]:
                    if f in VALID_ADDRESSES[k]:
                        self.addresses[k].append(f)
                    else:
                        raise Warning('{} is not a valid feature for {}. '.format(f, k) +
                                      'Please pick from this list : {}'.format(VALID_ADDRESSES[k]))
            else:
                raise Warning('{} is not a valid name. Please pick from this list : {}'.format(k, VALID_ADDRESSES))

    def apply_action(self, network_batch, y):
        for k in y.keys():
            for f in y[k].keys():
                y_batch = np.transpose(y[k][f], [0,2,1])
                for network_window, y_window in zip(network_batch, y_batch):
                    for network_sample, y_sample in zip(network_window, y_window):
                        if k == 'bus':
                            df = pd.DataFrame(data=y_sample,
                                              index=network_sample.get_buses().index,
                                              columns=[f])
                            network_sample.update_buses(df)
                        elif k == 'gen':
                            df = pd.DataFrame(data=y_sample,
                                              index=network_sample.get_generators().index,
                                              columns=[f])
                            network_sample.update_generators(df)
                        elif k == 'load':
                            df = pd.DataFrame(data=y_sample,
                                              index=network_sample.get_loads().index,
                                              columns=[f])
                            network_sample.update_loads(df)
                        elif k == 'line':
                            df = pd.DataFrame(data=y_sample,
                                              index=network_sample.get_loads().index,
                                              columns=[f])
                            network_sample.update_loads(df)
                        elif k == 'twt':
                            df = pd.DataFrame(data=y_sample,
                                              index=network_sample.get_loads().index,
                                              columns=[f])
                            network_sample.update_loads(df)
                        else:
                            raise ValueError('Object {} is not a valid object name. ' +
                                             'Please pick from this list : {}'.format(k, VALID_FEATURES))

    def compute_load_flow(self, network_batch):
        for network_window in network_batch:
            for network_sample in network_window:
                pl.run_ac(network_sample)

    def get_features(self, network_batch, features, flat=False):
        r = {}
        for k in features.keys():
            r[k] = {}
            for f in features[k]:
                r[k][f] = []
                for network_window in network_batch:
                    temp = []
                    for network_sample in network_window:
                        if k == 'bus':
                            temp.append(network_sample.get_buses()[f])
                        elif k == 'gen':
                            temp.append(network_sample.get_generators()[f])
                        elif k == 'load':
                            temp.append(network_sample.get_loads()[f])
                        elif k == 'line':
                            temp.append(network_sample.get_lines()[f])
                        elif k == 'twt':
                            temp.append(network_sample.get_lines()[f])
                        else:
                            raise ValueError('Object {} is not a valid object name. ' +
                                             'Please pick from this list : {}'.format(k, VALID_FEATURES))
                    temp = np.array(temp)
                    r[k][f].append(temp)
                r[k][f] = np.stack(r[k][f], axis=0)
                r[k][f] = np.transpose(r[k][f], [0, 2, 1])
        if flat:
            r = self.flatten_features(r)
        return r

    def flatten_features(self, y):
        y_flat = {}
        for k in y.keys():
            y_flat[k] = {}
            for f in y[k].keys():
                n = y[k][f].shape[-1]
                y_flat[k][f] = np.reshape(y[k][f], [-1, n])
        return y_flat

