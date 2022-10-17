import torch
import numpy as np
from torch.utils.data import Dataset


class PowerGridDataset(Dataset):
    """Subclass of torch.utils.data.Dataset that supports our data formalism.

    It returns a pair `(x, net)` where :

        1. `a` describes the addresses of the different objects that are present in the power grid ;
        2. `x` describes the numerical features of the different ovjects that are present in the power grid ;
        3. `net` is a power grid instance based on the backend. This part of the data will serve if one needs
            to interact with the said power grid, by performing AC Power Flow simulations for instance.

    """

    def __init__(self, data_dir=None, backend=None, **kwargs):
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
        self.backend = backend
        self.data_structure = kwargs.get('data_structure', self.backend.valid_data_structure)
        self.backend.check_data_structure(self.data_structure)
        self.files = self.backend.get_valid_files(data_dir)
        self.normalizer = kwargs.get('normalizer', None)
        self.transform = kwargs.get('transform', None)
        # self.address_names = kwargs.get('address_names', self.backend.valid_address_names)
        # self.backend.check_address_names(self.address_names)
        # self.feature_names = kwargs.get('feature_names', self.backend.valid_feature_names)
        # self.backend.check_feature_names(self.feature_names)

    def __getitem__(self, index):
        """Returns a tuple `(a, x, net)`."""
        filename = self.files[index]
        net = self.backend.load_network(filename)
        x = self.backend.get_data_network(net, self.data_structure)
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
