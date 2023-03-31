from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple
import tqdm
from gymnasium import Env, spaces
from ml4ps.h2mg import H2MG, H2MGSpace, collate_h2mgs
from torch.utils.data import Dataset, DataLoader
import os


class PSBasePb(ABC):
    """
    """

    def __init__(self, data_dir: str, batch_size: int = 8, shuffle: bool = True, load_in_memory: bool = True):
        """
            Initialize problem
        """
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.load_in_memory = load_in_memory

        path = os.path.join(data_dir, self.__class__.__name__)
        if not os.path.isdir(path):
            os.mkdir(path)

        input_structure_name = os.path.join(path, 'input_structure.pkl')
        self.input_structure = self.backend.get_max_structure(input_structure_name, self.data_dir, self.empty_input_structure)

        output_structure_name = os.path.join(path, 'output_structure.pkl')
        self.output_structure = self.backend.get_max_structure(output_structure_name, self.data_dir, self.empty_output_structure)

        self.input_space = H2MGSpace.from_structure(self.input_structure)
        self.output_space = self._build_output_space(self.output_structure)

        self.data_set = PSDataset(data_dir=self.data_dir,
                                  backend=self.backend,
                                  input_structure=self.input_structure,
                                  output_structure=self.output_structure,
                                  load_in_memory=self.load_in_memory)

        def collate_fn(list):
            x_list = [a[0] for a in list]
            y_list = [a[1] for a in list]
            return collate_h2mgs(x_list), collate_h2mgs(y_list)

        self.data_loader = DataLoader(self.data_set, batch_size=self.batch_size,
                                      shuffle=self.shuffle, collate_fn=collate_fn)

    def __iter__(self):
        return iter(self.data_loader)

    @property
    @abstractmethod
    def backend(self):
        pass

    @property
    @abstractmethod
    def empty_input_structure(self):
        pass

    @property
    @abstractmethod
    def empty_output_structure(self):
        pass

    @abstractmethod
    def _build_output_space(self, output_structure):
        pass


class PSDataset(Dataset):
    def __init__(self, data_dir=None, backend=None, input_structure=None, output_structure=None, load_in_memory=False):
        self.data_dir = data_dir
        self.backend = backend

        self.input_structure = input_structure
        self.output_structure = output_structure
        self.load_in_memory = load_in_memory
        self.files = self.backend.get_valid_files(data_dir)

        # If asked, load all the dataset in memory
        if self.load_in_memory:
            self.dataset = []
            for index in tqdm.tqdm(range(len(self.files)), desc='Loading the dataset in memory.'):
                data = self._load_item(index)
                self.dataset.append(data)

    def __getitem__(self, index):
        """"""
        if self.load_in_memory:
            return self._get_item_from_memory(index)
        else:
            return self._load_item(index)

    def _get_item_from_memory(self, index):
        return self.dataset[index]

    def _load_item(self, index: int):
        filename = self.files[index]
        power_grid = self.backend.load_power_grid(filename)
        x = self.backend.get_h2mg_from_power_grid(power_grid, structure=self.input_structure)
        y = self.backend.get_h2mg_from_power_grid(power_grid, structure=self.output_structure)
        return x, y

    def __len__(self):
        """Length of the dataset."""
        return len(self.files)
