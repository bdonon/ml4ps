from ml4ps.utils import collate_dict, separate_dict, assert_substructure
from abc import ABC, abstractmethod
import numpy as np
import os


class AbstractBackend(ABC):
    """Abstract Power Systems backend.

        Allows to load power grids, get and set features, and to interact with them through Power Flow simulations.

        Attributes:
            valid_extensions (:obj:`list` of :obj:`str`): List of valid file extensions that can be read by the
                backend. Should be overridden in a proper backend implementation.
            valid_object_names (:obj:`list` of :obj:`str`): List of object names.
            valid_address_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that contains all the valid
                object names as keys and valid address names for each of these keys. Should be overridden in a
                proper backend implementation.
            valid_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that contains all the valid
                object names as keys and valid feature names for each of these keys. Should be overridden in a
                proper backend implementation.
    """

    def __init__(self):
        """Initializes a Power Systems backend."""
        pass

    @property
    @abstractmethod
    def valid_extensions(self):
        pass

    @property
    @abstractmethod
    def valid_address_names(self):
        pass

    @property
    @abstractmethod
    def valid_feature_names(self):
        pass

    @abstractmethod
    def load_network(self, file_path):
        """Loads a single power grid instance.

        Should be overridden in a proper backend implementation.
        Should be consistent with `valid_extensions`.
        """
        pass

    @abstractmethod
    def save_network(self, net, path):
        """Saves a single power grid instance in path.

        Should be overridden in a proper backend implementation.
        """
        pass

    def save_batch(self, network_batch, path):
        """Saves a batch of power grid instances in path."""
        [self.save_network(net, path) for net in network_batch]

    @abstractmethod
    def set_data_network(self, net, y):
        """Modifies a power grid with the feature values contained in y.

        Should be overridden in a proper backend implementation.
        Should be consistent with `valid_feature_names`.
        """
        pass

    def set_data_batch(self, network_batch, y_batch):
        """Modifies a batch of power grids with a batch of features."""
        [self.set_data_network(network, y) for network, y in zip(network_batch, separate_dict(y_batch))]

    @abstractmethod
    def run_network(self, net, **kwargs):
        """Performs a single power flow computation.

        Should be overridden in a proper backend implementation.
        """
        pass

    def run_batch(self, network_batch, **kwargs):
        """Performs power flow computations for a batch of power grids."""
        [self.run_network(net, **kwargs) for net in network_batch]

    @abstractmethod
    def get_data_network(self, net, feature_names=None, address_names=None, address_to_int=True):
        """Returns feature values from a single power grid instance.

        Should be overridden in a proper backend implementation.
        Should be consistent with `valid_data_structure`.
        """
        pass

    def get_data_batch(self, net_batch, **kwargs):#feature_names=None, address_names=None, address_to_int=True):
        """Returns features from a batch of power grids."""
        return collate_dict([self.get_data_network(net, **kwargs) for net in net_batch])

    def assert_names(self, feature_names=None, address_names=None):
        """Asserts that `object_names`, `feature_names` and `address_names` are valid w.r.t. the backend."""
        if feature_names is not None:
            assert_substructure(feature_names, self.valid_feature_names)
        if address_names is not None:
            assert_substructure(address_names, self.valid_address_names)

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
