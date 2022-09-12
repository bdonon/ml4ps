"""Interfaces allow to interact with a dataset of power grids.

An interface allows to iterate over the train, validation and test sets.
It returns addresses, features and networks that are in a format
that is specific to the used backend. It also allows to retrieve features
from networks, to modify those networks (using the output of a neural
network for instance), and to perform power flow computations.

  Typical usage example:

  my_interface = Interface()
  for a, x, net_batch in my_interface.train:
  y_dict = my_neural_network(a, x)
  my_interface.update_network_batch(network_batch, y_dict)
  my_interface.run_load_flow_batch(network_batch)
  z = my_interface.get_features_dict(network_batch, features)

"""
from ML4PS.backend.pypowsybl import Backend
from ML4PS.backend.pandapower import Backend
#from ML4PS.backend.interface import get_backend
from ML4PS.iterator import Iterator
import math
import os
from tqdm import tqdm

class Interface:
    """Interfaces allow to interact with a dataset of power grids.

    An interface allows to iterate over the train, validation and test sets.
    It returns addresses, features and networks that are in a format
    that is specific to the used backend. It also allows to retrieve features
    from networks, to modify those networks (using the output of a neural
    network for instance), and to perform power flow computations.

    Attributes:
        backend_name: A string indicating the desired backend.
        backend: A backend implementation that inherits from AbstractBackend.
        series_length: An integer that defines the coherence length of
            time series.
        validation_portion: A list of 2 integers that define the proportions of
            the start and end of the validation set (e.g. [0.9, 1.0] will keep
            the last 10% of the train set as validation set).
        data_dir: A string defining the path to a dataset directory that must
            contain a train and test directories, each containing files with
            a valid extension w.r.t. the backend.
        all_train_files: A list of strings of all the valid files contained
            in the train directory.
        train_files: A list of strings of all the files kept for the train set.
        val_files: A list of strings of all the files kept for the val set.
        test_files: A list of strings of all the valid files contained
            in the test directory.
        train: An iterator to iterate through the train set.
        val: An iterator to iterate through the val set.
        test: An iterator to iterate through the test set.

    """

    def __init__(self, **kwargs):
        """Inits interface."""

        self.backend_name = kwargs.get("backend_name", "pypowsybl")
        self.series_length = kwargs.get("series_length", 1)
        self.validation_portion = kwargs.get("validation_portion", [0.9, 1.0])
        self.data_dir = kwargs.get("data_dir", None)

        self.backend = get_backend(self.backend_name)
        # if self.backend_name == 'pypowsybl':
        #    self.backend = PyPowSyblBackend()
        # elif self.backend_name == 'pandapower':
        #    self.backend = PandaPowerBackend()
        # else:
        #    raise ValueError('Not a valid backend !')

        # Build train, val and test lists of valid files
        self.all_train_files = self.backend.get_files(os.path.join(self.data_dir, 'train'))
        self.train_files, self.val_files = self.split_train_val(self.all_train_files)
        self.test_files = self.backend.get_files(os.path.join(self.data_dir, 'test'))

        # Make sure that there is at least one valid file in all sets
        if (not self.train_files) or (not self.val_files) or (not self.test_files):
            raise FileNotFoundError("There is no valid file in {}".format(self.data_dir))

        self.train = Iterator(self.train_files, self.backend, **kwargs)
        self.val = Iterator(self.val_files, self.backend, **kwargs)
        self.test = Iterator(self.test_files, self.backend, **kwargs)

    def split_train_val(self, files):
        """Split train and validation sets, while respecting time windows."""
        val_start_id = math.floor(self.validation_portion[0] * len(files) / self.series_length) * self.series_length
        val_end_id = math.ceil(self.validation_portion[1] * len(files) / self.series_length) * self.series_length
        val_files = files[val_start_id:val_end_id]
        train_files = files[:val_start_id] + files[val_end_id:]
        return train_files, val_files

    def update_network_batch(self, network_batch, y_concat):
        """Updates a batch of power networks using values contained in dict y_concat."""
        y_batch = self.backend.concat_to_batch(y_concat)
        self.backend.update_network_batch(network_batch, y_batch)

    def run_load_flow_batch(self, network_batch, load_flow_options=None):
        """Performs load flow computations for a batch of power networks."""
        self.backend.run_load_flow_batch(network_batch, load_flow_options)

    def get_features_dict(self, network_batch, features):
        """Gets features defined in features from a batch of power networks"""
        y_batch = self.backend.extract_feature_batch(network_batch, features)
        return self.backend.batch_to_concat(y_batch)

    def get_train_batch(self):
        return tqdm(self.train)

    def get_val_batch(self):
        return tqdm(self.val)

    def get_test_batch(self):
        return tqdm(self.test)
