import torch
import numpy as np
from torch.utils.data import Dataset


def power_grid_collate(data):
    """Collate function to pass to a torch.utils.data.DataLoader.

    It collates addresses `a` and features `x` using the default collate function of torch,
    but keeps the network list untouched.

    """
    a, x, network = zip(*data)
    a = torch.utils.data.default_collate(a)
    a = {k: {f: np.array(a[k][f]) for f in a[k].keys()} for k in a.keys()}
    x = torch.utils.data.default_collate(x)
    x = {k: {f: np.array(x[k][f]) for f in x[k].keys()} for k in x.keys()}
    return a, x, network


class PowerGridDataset(Dataset):
    """Subclass of torch.utils.data.Dataset that supports our data formalism.

    It returns a tuple `(a, x, net)` where :

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
        self.files = self.backend.get_files(data_dir)
        self.normalizer = kwargs.get('normalizer', None)
        self.addresses = kwargs.get('addresses', self.backend.valid_addresses)
        self.backend.check_addresses(self.addresses)
        self.features = kwargs.get('features', self.backend.valid_features)
        self.backend.check_features(self.features)

    def __getitem__(self, index):
        """Returns a tuple `(a, x, net)`."""
        filename = self.files[index]
        net = self.backend.load_network(filename)
        a = self.backend.extract_addresses(net, self.addresses)
        x = self.backend.extract_features(net, self.features)
        if self.normalizer is not None:
            x = self.normalizer(x)
        return a, x, net

    def __len__(self):
        """Length of the dataset."""
        return len(self.files)
