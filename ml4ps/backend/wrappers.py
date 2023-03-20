import json
import tqdm
import os
import numpy as np
from ml4ps.backend.interface import AbstractBackend
from ml4ps.h2mg import H2MGStructure
import pickle


def get_max_structure(backend, data_dir, empty_structure):
    max_structure = H2MGStructure()
    valid_files = backend.get_valid_files(data_dir)
    for file in tqdm.tqdm(valid_files, desc='Counting maximal class count in {}'.format(data_dir)):
        power_grid = backend.load_power_grid(file)
        h2mg = backend.get_h2mg_from_power_grid(power_grid, structure=empty_structure)
        structure = h2mg.structure
        max_structure = max_structure.max(structure)
    return max_structure


class PaddingWrapper(AbstractBackend):
    """Backend wrapper that pads samples of a dataset to have constant dimensions."""

    def __init__(self, backend, structure=None):
        """Inits a padding wrapper."""
        super().__init__(structure)
        self.backend = backend

    @property
    def valid_structure(self):
        return self.backend.valid_structure

    @property
    def valid_extensions(self):
        return self.backend.valid_extensions

    @property
    def valid_global_feature_names(self):
        return self.backend.valid_global_feature_names

    def load_power_grid(self, file_path, **kwargs):
        """Loads a power grid after ensuring that it belongs to the dataset considered by the wrapper."""
        return self.backend.load_power_grid(file_path, **kwargs)

    def save_power_grid(self, power_grid, path, **kwargs):
        """Saves a power grid instance."""
        self.backend.save_power_grid(power_grid, path, **kwargs)

    def run_power_grid(self, power_grid, **kwargs):
        """Runs a power grid simulation."""
        self.backend.run_power_grid(power_grid, **kwargs)

    def get_h2mg_from_power_grid(self, power_grid, structure=None, **kwargs):
        """Pads the output of `backend.get_data_power_grid` to be consistent with the dataset max amount of objects."""
        h2mg = self.backend.get_h2mg_from_power_grid(power_grid, structure=structure, **kwargs)
        return h2mg.pad_with_nans(structure)

    def set_h2mg_into_power_grid(self, power_grid, h2mg, **kwargs):
        """Unpads data contained in `y` to get rid of fictitious objects and apply it to the power grid instance."""
        h2mg_unpadded = h2mg.unpad_nans()
        self.backend.set_h2mg_into_power_grid(power_grid, h2mg_unpadded, **kwargs)

    def get_max_structure(self, name, data_dir, empty_structure):
        """Gets the maximal structure from `data_dir`, following the template `empty_structure`"""
        if os.path.isfile(name):
            with open(name, 'rb') as f:
                max_structure = pickle.load(f)
        else:
            max_structure = get_max_structure(self.backend, data_dir, empty_structure)
            with open(name, 'wb') as f:
                pickle.dump(max_structure, f)
        return max_structure
