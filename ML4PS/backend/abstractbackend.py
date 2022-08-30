from abc import ABC, abstractmethod
import numpy as np


class AbstractBackend(ABC):

    def __init__(self):
        pass

    @property
    @abstractmethod
    def valid_extensions(self):
        pass

    @property
    @abstractmethod
    def valid_addresses(self):
        pass

    @property
    @abstractmethod
    def valid_features(self):
        pass

    def get_table_dict_batch(self, network_batch, features):
        return [[self.get_table_dict(ns, features) for ns in nw] for nw in network_batch]

    def get_table_dict(self, network, features):
        return {k: self.get_table(network, k) for k in features.keys()}

    @abstractmethod
    def get_table(self, net, key):
        pass

    def load_network_batch(self, batch_files):
        return [[self.load_network(file) for file in window_files] for window_files in batch_files]

    @abstractmethod
    def load_network(self, file_path):
        pass

    def update_network_batch(self, network_batch, y_batch):
        for network_window, y_window in zip(network_batch, y_batch):
            for network_sample, y_sample in zip(network_window, y_window):
                self.update_network(network_sample, y_sample)

    @abstractmethod
    def update_network(self, net, y):
        pass

    def run_load_flow_batch(self, network_batch, load_flow_options=None):
        for network_window in network_batch:
            for network_sample in network_window:
                self.run_load_flow(network_sample, load_flow_options)

    @abstractmethod
    def run_load_flow(self, net, load_flow_options=None):
        pass

    def extract_feature_batch(self, network_batch, features):
        table_dict_batch = self.get_table_dict_batch(network_batch, features)
        return [[self.extract_features(tds, features) for tds in tdw] for tdw in table_dict_batch]

    def extract_features(self, table_dict, features):
        r = {}
        for k, v in features.items():
            r[k] = {}
            for f in v:
                if f in table_dict[k].keys():
                    r[k][f] = table_dict[k][f].astype(float).to_numpy()
        return r
        #return {k: {f: table_dict[k][f].astype(float).to_numpy() for f in v} for k, v in features.items()}

    def extract_address_batch(self, network_batch, addresses):
        table_dict_batch = self.get_table_dict_batch(network_batch, addresses)
        return [[self.extract_addresses(tds, addresses) for tds in tdw] for tdw in table_dict_batch]

    def extract_addresses(self, table_dict, addresses):
        address_to_id_dict = self.get_unique_id_dict(table_dict, addresses)
        return {k: {f: table_dict[k][f].astype(str).map(address_to_id_dict) for f in v} for k, v in
                addresses.items()}

    def get_unique_id_dict(self, table_dict, addresses):
        all_addresses = [list(table_dict[k][f].values.astype(str)) for k, v in addresses.items() for f in v]
        unique_addresses = list(np.unique(np.concatenate(all_addresses)))
        return {address: i for i, address in enumerate(unique_addresses)}

    def batch_to_concat(self, v_batch, address=False):
        # TODO : One should be able to concatenate networks of varying size
        v_concat = {k: {f: [] for f in v_batch[0][0][k].keys()} for k in v_batch[0][0].keys()}
        for k in v_batch[0][0].keys():
            for f in v_batch[0][0][k].keys():
                for v_window in v_batch:
                    if address:
                        temp = [v_window[0][k][f]]
                    else:
                        temp = []
                        for v_sample in v_window:
                            temp.append(v_sample[k][f])
                    temp = np.array(temp)
                    v_concat[k][f].append(temp)
                v_concat[k][f] = np.array(v_concat[k][f])
                v_concat[k][f] = np.transpose(v_concat[k][f], [0,2,1])
        return v_concat

    def concat_to_batch(self, v_concat):
        elem = list(list(v_concat.values())[0].values())[0]
        batch_size = np.shape(elem)[0]
        window_size = np.shape(elem)[2]
        v_batch = []
        for i in range(batch_size):
            v_window = []
            for j in range(window_size):
                v_sample = {k: {f: v_concat[k][f][i,:,j] for f in v_concat[k].keys()} for k in v_concat.keys()}
                v_window.append(v_sample)
            v_batch.append(v_window)
        return v_batch

    def check_features(self, features):
        for k in features.keys():
            if k in self.valid_features.keys():
                for f in features[k]:
                    if f in self.valid_features[k]:
                        continue
                    else:
                        raise Warning('{} is not a valid feature for {}. '.format(f, k) +
                                      'Please pick from this list : {}'.format(self.valid_features[k]))
            else:
                raise Warning('{} is not a valid name. Please pick from this list : {}'.format(k, self.valid_features))

    def check_addresses(self, addresses):
        for k in addresses.keys():
            if k in self.valid_addresses.keys():
                for f in addresses[k]:
                    if f in self.valid_addresses[k]:
                        continue
                    else:
                        raise Warning('{} is not a valid feature for {}. '.format(f, k) +
                                      'Please pick from this list : {}'.format(self.valid_addresses[k]))
            else:
                raise Warning('{} is not a valid name. Please pick from this list : {}'.format(k, self.valid_addresses))

    def clean_dict(self, v):
        keys_to_erase = []
        for k, v_k in v.items():
            keys_to_erase_k = []
            for f, v_k_f in v_k.items():
                if np.prod(np.shape(v_k_f)) == 0:
                    keys_to_erase_k.append(f)
            for f in keys_to_erase_k:
                del v_k[f]
            if not v_k:
                keys_to_erase.append(k)
        for k in keys_to_erase:
            del v[k]