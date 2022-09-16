from abc import ABC, abstractmethod
import numpy as np
import os


def collate_dict(data):
    """Transforms a list of dictionaries into a dictionary whose values are tensors with an additional dimension."""
    return {k: {f: np.array([d[k][f] for d in data]) for f in data[0][k].keys()} for k in data[0].keys()}


def collate_power_grid(data):
    """Collates tuples `(a, x, nets)`, by only collating `a` and `x` and leaving `nets` untouched."""
    a, x, network = zip(*data)
    return collate_dict(a), collate_dict(x), network


def separate_dict(data):
    """Transforms a dict of batched tensors into a list of dicts that have single tensors as values."""
    elem = list(list(data.values())[0].values())[0]
    batch_size = np.shape(elem)[0]
    return [{k: {f: data[k][f][i] for f in v} for k, v in data.items()} for i in range(batch_size)]




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

    @abstractmethod
    def load_network(self, file_path):
        """Loads a single power grid instance."""
        pass

    def set_feature_batch(self, network_batch, y_batch):
        """Modifies a batch of power grids with a batch of features."""
        [self.set_feature_network(network, y) for network, y in zip(network_batch, separate_dict(y_batch))]

    @abstractmethod
    def set_feature_network(self, net, y):
        """Modifies a power grid with the feature values contained in y."""
        pass

    def run_batch(self, network_batch, **kwargs):
        """Performs power flow computations for a batch of power grids."""
        [self.run_network(net, **kwargs) for net in network_batch]

    @abstractmethod
    def run_network(self, net, **kwargs):
        """Performs a single power flow computation."""
        pass

    def get_feature_batch(self, network_batch, features):
        """Returns features from a batch of power grids."""
        return collate_dict([self.get_feature_network(network, features) for network in network_batch])

    @abstractmethod
    def get_feature_network(self, network, features):
        """Returns features from a single power grid instance."""
        pass

    @abstractmethod
    def get_address_network(self, network, addresses):
        """Extracts a nested dict of address ids from a power grid instance. Should return integers."""
        pass

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

    def get_valid_files(self, path, shuffle=False, n_samples=None):
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
