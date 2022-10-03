from ml4ps.backend.interface import AbstractBackend

import json

class NoBackend(AbstractBackend):
    """ A way to to load a `.json` file without using a backend, it assumes the follow structure:
        {'elem1': {'feat1: [...], 'feat2: [...]}, 'elem2': {'feat1: [...], 'feat2: [...]}}
    """

    valid_extensions = (".json")
    valid_feature_names = []
    valid_address_names = []

    def __init__(self):
        """Initializes a NoBackend."""
        super().__init__()
        
    def warning(self):
        print("No backend is used")
        
    def warning_none_return(self):
        self.warning()
        return None
        
    def load_network(self, file_path):
        """Loads data stored in a `.json` file."""
        with open(file_path, encoding = 'utf-8') as f:
            return json.load(f)

    def set_feature_network(self, net, y):
        """Updates a power grid by setting features according to `y`."""
        print("No backends are used")


    def run_network(self, net, **kwargs):
        """Send a warning"""
        self.warning()

    def get_feature_network(self, network, feature_names):
        """Returns features from a single power grid instance."""
        feat = {}
        for elem in feature_names.keys():
            feat[elem] = {}
            for f in feature_names[elem]:
                feat[elem][f] = network[elem][f]
        return feat
    
    def check_feature_names(self, feature_names):
        """Checks are bypassed."""
        pass
    
    def check_address_names(self, feature_names):
        """Checks are bypassed."""
        pass

    def get_address_network(self, network, address_names):
        """Extracts a nested dict of address ids from a power grid instance."""
        adr = {}
        for elem in address_names.keys():
            adr[elem] = {}
            for a in address_names[elem]:
                adr[elem][a] = network[elem][a]
        return adr
