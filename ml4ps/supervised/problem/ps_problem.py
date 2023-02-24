from abc import ABC, abstractmethod
from typing import Any, Optional, Dict, Tuple
import tqdm
from gymnasium import Env, spaces
from ml4ps.h2mg.core import H2MG, collate_h2mgs
from torch.utils.data import Dataset, DataLoader
from ml4ps.reinforcement.h2mg_space import H2MGSpace


class PSBasePb(ABC):
    """
        Power system base supervised learning problem.

        Attributes:
            data_dir: str
            backend: Any
            address_names: Dict[List]
            input_var_names: Dict[List]
            input_space: spaces.Space
            output_var_names: Dict[List]
            output_space: spaces.Space
    """


    data_dir: str
    backend: Any
    global_input_feature_names: Dict
    local_input_feature_names: Dict
    local_address_names: Dict
    global_output_feature_names: Dict
    local_output_feature_names: Dict
    output_space: H2MGSpace
    n_obj: dict
    batch_size: int = 1
    shuffle: bool = True
    load_in_memory: bool = False

    def __init__(self):
        """
            Initialize problem
        """
        super().__init__()
        self._n_obj = self.backend.max_n_obj.copy()
        self.data_set = PSDataset(data_dir=self.data_dir,
                                  backend=self.backend,
                                  global_input_feature_names=self.global_input_feature_names,
                                  local_input_feature_names=self.local_input_feature_names,
                                  local_address_names=self.local_address_names,
                                  global_output_feature_names=self.global_output_feature_names,
                                  local_output_feature_names=self.local_output_feature_names,
                                  load_in_memory=self.load_in_memory)
        if self.batch_size == 1:
            self.data_loader = DataLoader(self.data_set, batch_size=self.batch_size, shuffle=self.shuffle,
                                           collate_fn=lambda xy: (xy[0][0], xy[0][1]))
        else:
            def collate_fn(list):
                x_list = [a[0] for a in list]
                y_list = [a[1] for a in list]
                return collate_h2mgs(x_list), collate_h2mgs(y_list)

            self.data_loader = DataLoader(self.data_set, batch_size=self.batch_size, shuffle=self.shuffle,
                                          collate_fn=collate_fn)
        # TODO : vérifier que le collate fonctionne, et qu'il renvoie bien des h2mg ?

    def __iter__(self):
        return iter(self.data_loader)


class PSDataset(Dataset):
    def __init__(self, data_dir=None, backend=None, global_input_feature_names=None, local_input_feature_names=None,
                 local_address_names=None, global_output_feature_names=None, local_output_feature_names=None,
                 load_in_memory=False):
        self.data_dir = data_dir
        self.backend = backend

        self.global_input_feature_names = global_input_feature_names
        self.local_input_feature_names = local_input_feature_names
        self.local_address_names = local_address_names
        self.global_output_feature_names = global_output_feature_names
        self.local_output_feature_names = local_output_feature_names
        #self.backend.assert_names(feature_names=self.feature_names, address_names=self.address_names)
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

    def _load_item(self, index):
        filename = self.files[index]
        power_grid = self.backend.load_power_grid(filename)
        x = self.backend.get_data_power_grid(power_grid,
                                          global_feature_names=self.global_input_feature_names,
                                          local_feature_names=self.local_input_feature_names,
                                          local_address_names=self.local_address_names)
        y = self.backend.get_data_power_grid(power_grid,
                                          global_feature_names=self.global_output_feature_names,
                                          local_feature_names=self.local_output_feature_names)
        return H2MG(x), H2MG(y)

    def __len__(self):
        """Length of the dataset."""
        return len(self.files)