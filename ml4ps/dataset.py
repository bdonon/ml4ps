import os
import pickle
import torch
import numpy as np
from torch.utils.data import Dataset
import pandapower as pp
import tqdm


class PowerGridDataset(Dataset):
    """Subclass of torch.utils.data.Dataset that supports our data formalism.

    It returns a pair `(x, net)` where `x` describes the numerical features and addresses of the objects that compose
    the power grid, and `net` is a power grid instance.
    """

    def __init__(self, data_dir=None, backend=None, pickle=False, return_network=True, load_in_memory=False, **kwargs):
        """Initializes a power grid dataset.

        Args:
            data_dir (:obj:`str`): Directory where the dataset is stored.
            backend (:obj:`ml4ps.backend.interface.Backend`): Power Systems backend that can load power grid
                instances, get and set power grid features, and perform Power Flow simulations.
            pickle (:obj:`bool`): States if the data_dir contains a picklized version of the dataset.
            return_network (:obj:`bool`): If true, the dataset returns pairs (x, net). If false, it only returns x.
            load_in_memory (:obj:`bool`): If true, the dataset is preloaded in memory. Beware of large datasets
                that may not fit in memory.
            normalizer (:obj:`ml4ps.normalization.Normalizer`, optional): Feature normalizer that can transform
                raw data (that is very likely to follow a multimodal distribution) into a distribution that is
                more suitable for the training of neural networks. If not provided, no normalization is applied
                to the features.
            addresses (:obj:`dict` of :obj:`list` of :obj:`str`, optional): Defines the addresses that the Dataset
                should return. The object names and address names should be compatible with the provided backend.
                If no addresses is provided, then the Dataset simply uses the default objects and addresses
                provided by the backend.
            features (:obj:`dict` of :obj:`list` of :obj:`str`, optional): Defines the features that the Dataset
                should return. The object names and feature names should be compatible with the provided backend.
                If no addresses is provided, then the Dataset simply uses the default objects and features
                provided by the ``backend.

        """
        self.data_dir = data_dir
        self.backend = backend

        self.feature_names = kwargs.get('feature_names', self.backend.valid_feature_names)
        self.address_names = kwargs.get('address_names', self.backend.valid_address_names)
        self.backend.assert_names(feature_names=self.feature_names, address_names=self.address_names)
        self.pickle = pickle
        self.return_network = return_network
        self.load_in_memory = load_in_memory
        if self.pickle:
            self.files = []
            for file in os.listdir(data_dir):
                if file.endswith(".pkl"):
                    self.files.append(os.path.join(data_dir, file))
        else:
            self.files = self.backend.get_valid_files(data_dir)
        
        self.normalizer = kwargs.get('normalizer', None)
        #self.transform = kwargs.get('transform', None)
        # self.address_names = kwargs.get('address_names', self.backend.valid_address_names)
        # self.backend.check_address_names(self.address_names)
        # self.feature_names = kwargs.get('feature_names', self.backend.valid_feature_names)
        # self.backend.check_feature_names(self.feature_names)

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
        if self.pickle:
            with open(filename, 'rb') as fp:
                data = pickle.load(fp)
            if self.return_network:
                x, net = data['x'], data['net']
            else:
                x = data['x']
        else:
            net = self.backend.load_network(filename)
            x = self.backend.get_data_network(net, self.feature_names, self.address_names)


        if self.normalizer is not None:
            x = self.normalizer(x)

        #if self.transform is not None:
        #    x, net = self.transform(x, net)
        if self.return_network:
            return x, net
        else:
            return x

    def __len__(self):
        """Length of the dataset."""
        return len(self.files)
