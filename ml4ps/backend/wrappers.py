import json
import tqdm
import os
import numpy as np
from ml4ps.backend.interface import AbstractBackend


def pad(x, max_n_obj):
    """Pads features contained in `x` with `np.nan` to have the amount of objects specified in `max_n_obj`."""
    for k, v in x["local_features"].items():
        for f, w in v.items():
            if len(w) > max_n_obj.get(k, 0):
                raise ValueError("Padding impossible, maximum {} objects of class {} ".format(max_n_obj.get(k, 0), k) +\
                                 "are present in this dataset, while {} ".format(len(w)) +\
                                 "objects are present in this sample.")
    for k, v in x["local_addresses"].items():
        for f, w in v.items():
            if len(w) > max_n_obj.get(k, 0):
                raise ValueError("Padding impossible, maximum {} objects of class {} ".format(max_n_obj.get(k, 0), k) +\
                                 "are present in this dataset, while {} ".format(len(w)) +\
                                 "objects are present in this sample.")
    r = {}
    if "local_features" in x:
        r["local_features"] = {k: {f: np.concatenate([w, np.full(max_n_obj.get(k,0)-len(w), np.nan)]) for f, w in v.items()}
            for k, v in x["local_features"].items()}
    if "local_addresses" in x:
        r["local_addresses"] = {k: {f: np.concatenate([w, np.full(max_n_obj.get(k, 0) - len(w), np.nan)]) for f, w in v.items()} for k, v in
            x["local_addresses"].items()}
    if "all_addresses" in x:
        r["all_addresses"] = np.arange(max_n_obj["max_address"])
    if "global_features" in x:
        r["global_features"] = x["global_features"]
    return r

def unpad(x, current_n_obj):
    """Deletes features in x to correspond to fictitious objects according to dict `current_n_obj`."""
    r = {}
    if "local_features" in x:
        r["local_features"] = {}
        for k, v in x["local_features"].items():
            r["local_features"][k] = {}
            for f, w in v.items():
                if len(w) < current_n_obj.get(k, 0):
                    raise ValueError("Unpadding impossible, {} objects of class {} ".format(current_n_obj.get(k, 0), k) + \
                                     "are available in the power grid, while only {} ".format(len(w)) + \
                                     "numerical values are provided.")
                else:
                    r["local_features"][k][f] = w[:current_n_obj.get(k, 0)]
    if "local_addresses" in x:
        r["local_addresses"] = {}
        for k, v in x["local_addresses"].items():
            r["local_addresses"][k] = {}
            for f, w in v.items():
                if len(w) < current_n_obj.get(k, 0):
                    raise ValueError("Unpadding impossible, {} objects of class {} ".format(current_n_obj.get(k, 0), k) + \
                                     "are available in the power grid, while only {} ".format(len(w)) + \
                                     "numerical values are provided.")
                else:
                    r["local_addresses"][k][f] = w[:current_n_obj.get(k, 0)]
    if "all_addresses" in x:
        r["all_addresses"] = x["all_addresses"][:current_n_obj["max_address"]]
    if "global_features" in x:
        r["global_features"] = x["global_features"]
    return r

def count_max_n_obj(backend, data_dir):
    """Counts the max amount of objects of each class in the dataset located in `data_dir`, according to `backend`."""
    max_n_obj = dict()
    max_address = 0
    valid_files = backend.get_valid_files(data_dir)
    for file in tqdm.tqdm(valid_files, desc='Counting maximal amount of object of each class in {}'.format(data_dir)):
        power_grid = backend.load_power_grid(file)
        address_names = backend.valid_local_address_names
        x = backend.get_data_power_grid(power_grid, local_address_names={k: list(v) for k, v in address_names.items()})
        n_address = x["all_addresses"].shape[0]
        current_n_obj = {k: max([np.shape(w)[0] for f, w in v.items()]) for k, v in x["local_addresses"].items()}
        for k, v in current_n_obj.items():
            if v > max_n_obj.get(k, 0):
                max_n_obj[k] = v
        if n_address > max_address:
            max_address = n_address
    max_n_obj["max_address"] = max_address
    max_n_obj["global"] = 1
    return max_n_obj

class PaddingWrapper(AbstractBackend):
    """Backend wrapper that pads samples of a dataset to have constant dimensions.

    When initialized, it checks if there exists a file 'max_n_obj.json' in the `data_dir`. This file should contain
    the amount of object of each class in the dataset located in `data_dir`. If this file does not exist, then
    the wrapper proceeds to read the whole dataset, count the maximal amount of objects of each class and store
    this piece of information in a file 'max_n_obj.json'.

    When using `get_data_network`, the wrapper pads the extracted features with `np.nan` so that it has as many objects
    of each class as it is defined in max_n_obj. By doing so, all snapshots will return tensors that have the exact
    same dimensions.

    When using `set_data_network`, the wrapper gets rid of additional objects that were introduced by the padding,
    and then applies the values to the corresponding objects.

    To sum things up, this wrapper allows to pad a dataset that may contain snapshots that do not have the same
    amount of objects, thus making the tensor dimensions static from one snapshot to the other.

    Attributes:
        max_n_obj_file (:obj:`str`): Path to a file that contains the maximal amount of objects of each class.
        max_n_obj (:obj:`dict` of :obj:`int`): Dictionary of the maximal amount of objects of each class.
    """

    def __init__(self, backend, data_dir):
        """Inits a padding wrapper.

        Args:
            backend (:obj:`ml4ps.backend`): ml4ps backend implementation that should be wrapped.
            data_dir (:obj:`str`): Directory of a dataset of power grid files compatible with the backend.
        """
        super().__init__()
        self.backend = backend
        self.data_dir = data_dir
        self.max_n_obj_file = os.path.join(data_dir, self.backend.__class__.__name__+'_max_n_obj.pad')
        if os.path.isfile(self.max_n_obj_file):
            with open(self.max_n_obj_file) as f:
                self.max_n_obj = json.load(f)
        else:
            self.max_n_obj = count_max_n_obj(backend, data_dir)
            with open(self.max_n_obj_file, 'w') as f:
                json.dump(self.max_n_obj, f)

    @property
    def valid_extensions(self):
        return self.backend.valid_extensions

    @property
    def valid_local_address_names(self):
        return self.backend.valid_local_address_names

    @property
    def valid_local_feature_names(self):
        return self.backend.valid_local_feature_names

    @property
    def valid_global_feature_names(self):
        return self.backend.valid_global_feature_names

    def load_power_grid(self, file_path, **kwargs):
        """Loads a power grid after ensuring that it belongs to the dataset considered by the wrapper."""
        assert os.path.basename(file_path) in os.listdir(self.data_dir)
        return self.backend.load_power_grid(file_path, **kwargs)

    def save_power_grid(self, power_grid, path, **kwargs):
        """Saves a power grid instance."""
        self.backend.save_power_grid(power_grid, path, **kwargs)

    def run_power_grid(self, power_grid, **kwargs):
        """Runs a power grid simulation."""
        self.backend.run_power_grid(power_grid, **kwargs)

    def get_data_power_grid(self, power_grid, **kwargs):
        """Pads the output of `backend.get_data_power_grid` to be consistent with the dataset max amount of objects."""
        return pad(self.backend.get_data_power_grid(power_grid, **kwargs), self.max_n_obj)

    def set_data_power_grid(self, power_grid, y, **kwargs):
        """Unpads data contained in `y` to get rid of fictitious objects and apply it to the power grid instance."""
        y_old = self.backend.get_data_power_grid(power_grid, local_feature_names={k: list(v) for k, v in y["local_features"].items()})
        current_n_obj = {k: np.max([len(w) for f, w in v.items()]) for k, v in y_old["local_features"].items()}
        self.backend.set_data_power_grid(power_grid, unpad(y, current_n_obj), **kwargs)
