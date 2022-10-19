import torch
import numpy as np
from torch.utils.data import Dataset


class PowerGridDataset(Dataset):
    """Subclass of torch.utils.data.Dataset that supports our data formalism.

    It returns a pair `(x, net)` where `x` describes the numerical features and addresses of the objects that compose
    the power grid, and `net` is a power grid instance.
    """

    def __init__(self, data_dir, backend, **kwargs):
        """Initializes a power grid dataset.

        Args:
            data_dir (:obj:`str`): Directory where the dataset is stored.
            backend (:obj:`ml4ps.backend.interface.Backend`): Power Systems backend that can load power grid
                instances, get and set power grid features, and perform Power Flow simulations.
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
        self.files = self.backend.get_valid_files(data_dir)

        self.feature_names = kwargs.get('feature_names', self.backend.valid_feature_names)
        self.address_names = kwargs.get('address_names', self.backend.valid_address_names)
        self.backend.assert_names(feature_names=self.feature_names, address_names=self.address_names)
        self.normalizer = kwargs.get('normalizer', None)
        self.transform = kwargs.get('transform', None)
        # self.address_names = kwargs.get('address_names', self.backend.valid_address_names)
        # self.backend.check_address_names(self.address_names)
        # self.feature_names = kwargs.get('feature_names', self.backend.valid_feature_names)
        # self.backend.check_feature_names(self.feature_names)

    def __getitem__(self, index):
        """Returns a tuple `(x, net)`."""
        filename = self.files[index]
        net = self.backend.load_network(filename)
        x = self.backend.get_data_network(net, self.feature_names, self.address_names)
        #a = self.backend.get_address_network(net, self.address_names)
        #x = self.backend.get_feature_network(net, self.feature_names)

        if self.normalizer is not None:
            x = self.normalizer(x)

        if self.transform is not None:
            return self.transform(x, net)
        else:
            return x, net

    def __len__(self):
        """Length of the dataset."""
        return len(self.files)
