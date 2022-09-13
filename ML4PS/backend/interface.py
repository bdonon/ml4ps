from abc import ABC, abstractmethod
import numpy as np
import jax.numpy as jnp
import torch
import os


def clean_dict(v):
    """Cleans a dictionary of tensors by deleting keys whose values are empty."""
    keys_to_erase = []
    for k, v_k in v.items():
        keys_to_erase_k = []
        for f, v_k_f in v_k.items():
            if np.shape(v_k_f)[0] == 0:
                keys_to_erase_k.append(f)
        for f in keys_to_erase_k:
            del v_k[f]
        if not v_k:
            keys_to_erase.append(k)
    for k in keys_to_erase:
        del v[k]
    return v


def collate(data):
    """Transforms a list of dictionaries into a dictionary whose values are tensors with an additional dimension."""
    data = torch.utils.data.default_collate(data)
    return {k: {f: jnp.array(data[k][f]) for f in data[k].keys()} for k in data.keys()}


def separate(data):
    """Transforms a dict of batched tensors into a list of dicts that have single tensors as values."""
    elem = list(list(data.values())[0].values())[0]
    batch_size = np.shape(elem)[0]
    return [{k: {f: data[k][f][i] for f in v} for k, v in data.items()} for i in range(batch_size)]


def build_unique_id_dict(table_dict, addresses):
    """Builds a dictionary to convert `str` indices into unique `int`."""
    all_addresses = [list(table_dict[k][f].values.astype(str)) for k, v in addresses.items() for f in v]
    unique_addresses = list(np.unique(np.concatenate(all_addresses)))
    return {address: i for i, address in enumerate(unique_addresses)}


class AbstractBackend(ABC):
    """Abstract Power Systems backend.

        Allows to load power grids, get and set features, and to interact with them through Power Flow simulations.

        Attributes:
            valid_extensions (:obj:`list` of :obj:`str`): List of valid file extensions that can be read by the
                backend.
            valid_addresses (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that contains all the valid
                object names as keys and valid address names for each of these keys.
            valid_features (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that contains all the valid
                object names as keys and valid feature names for each of these keys.
    """

    def __init__(self):
        """Initializes a Power Systems backend."""
        pass

    @property
    @abstractmethod
    def valid_extensions(self):
        """List of valid file extensions that can be read by the backend."""
        pass

    @property
    @abstractmethod
    def valid_addresses(self):
        """Dictionary of keys that constitute valid addresses w.r.t. the backend."""
        pass

    @property
    @abstractmethod
    def valid_features(self):
        """Dictionary of keys that constitute valid features w.r.t. the backend."""
        pass

    def get_table_dict(self, network, features):
        """Gets a dict of pandas table for all the objects in features, from the input network."""
        return {k: self.get_table(network, k, f) for k, f in features.items()}

    @abstractmethod
    def get_table(self, net, key, feature_list):
        """Gets a pandas table that displays features corresponding to the object key, from the input network."""
        pass

    @abstractmethod
    def load_network(self, file_path):
        """Loads a single power grid instance."""
        pass

    def update_run_extract(self, network_batch, y_batch=None, features=None, load_flow=False, **kwargs):
        """Modifies a batch of power grids with features contained in y_batch."""
        # TODO rajouter du multiprocessing pour le load flow ?
        # TODO que faire en cas de divergence du load flow ?
        if y_batch is not None:
            y_batch = separate(y_batch)
            [self.update_network(net, y) for net, y in zip(network_batch, y_batch)]
        if load_flow:
            [self.run_load_flow(net, **kwargs) for net in network_batch]
        if features is not None:
            r = [self.extract_features(net, features) for net in network_batch]
            return collate(r)

    @abstractmethod
    def update_network(self, net, y):
        """Modifies a power grid with the feature values contained in y."""
        pass

    @abstractmethod
    def run_load_flow(self, net, **kwargs):
        """Performs a single power flow computation."""
        pass

    def extract_features(self, network, features):
        """Extracts a nested dict of feature values from a power grid instance."""
        table_dict = self.get_table_dict(network, features)
        x = {k: {f: table_dict[k][f].astype(float).to_numpy() for f in v} for k, v in table_dict.items()}
        return clean_dict(x)

    def extract_addresses(self, network, addresses):
        """Extracts a nested dict of address ids from a power grid instance."""
        table_dict = self.get_table_dict(network, addresses)
        id_dict = build_unique_id_dict(table_dict, addresses)
        a = {k: {f: table_dict[k][f].astype(str).map(id_dict).to_numpy() for f in v} for k, v in addresses.items()}
        return clean_dict(a)

    def check_features(self, features):
        """Checks that features are valid w.r.t. the current backend."""
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
        """Checks that addresses are valid w.r.t. the current backend."""
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


    def get_files(self, path, shuffle=False, n_samples=None):
        """Gets file that have a valid extension w.r.t. the backend, from path."""
        files = []
        for f in sorted(os.listdir(path)):
            if f.endswith(self.valid_extensions):
                files.append(os.path.join(path, f))
        if not files:
            raise FileNotFoundError("There is no valid file in {}".format(path))
        if shuffle:
            np.random.shuffle(files)
        if n_samples is not None:
            return files[:n_samples]
        else:
            return files
