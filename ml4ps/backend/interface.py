from abc import ABC, abstractmethod
import numpy as np
import os


class AbstractBackend(ABC):
    """Abstract Power Systems backend.

        Allows to load power grids, get and set features, and interact with them through Power Flow simulations.

        Attributes:
            valid_extensions (:obj:`list` of :obj:`str`): List of valid file extensions that can be read by the
                backend. Should be overridden in a proper backend implementation.
            valid_address_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that contains all the valid
                object names as keys and valid address names for each of these keys. Should be overridden in a
                proper backend implementation.
            valid_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that contains all the valid
                object names as keys and valid feature names for each of these keys. Should be overridden in a
                proper backend implementation.
    """

    def __init__(self, default_structure=None):
        """Initializes a Power Systems backend."""
        pass

    @property
    @abstractmethod
    def valid_extensions(self):
        pass

    @property
    @abstractmethod
    def valid_structure(self):
        pass

    @abstractmethod
    def load_power_grid(self, file_path):
        """Loads a single power grid instance.

        Should be overridden in a proper backend implementation.
        Should be consistent with `valid_extensions`.
        """
        pass

    @abstractmethod
    def save_power_grid(self, power_grid, path):
        """Saves a single power grid instance in path.

        Should be overridden in a proper backend implementation.
        """
        pass

    @abstractmethod
    def run_power_grid(self, power_grid):
        """Performs a single power flow computation.

        Should be overridden in a proper backend implementation.
        """
        pass

    @abstractmethod
    def set_h2mg_into_power_grid(self, power_grid, h2mg):
        """Modifies a power grid with the feature values contained in y.

        Should be overridden in a proper backend implementation.
        Should be consistent with `valid_feature_names`.
        """
        pass

    @abstractmethod
    def get_h2mg_from_power_grid(self, power_grid, structure=None, str_to_int=True):
        """Returns feature values from a single power grid instance.

        Should be overridden in a proper backend implementation.
        Should be consistent with `valid_data_structure`.
        """
        pass

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
