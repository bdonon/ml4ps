import pickle
from ml4ps.transform import get_transform


class PostProcessor:
    """Savable and loadable post processor that can be applied to the output of a neural network."""

    def __init__(self, filename=None, **kwargs):
        """Initializes a PostProcessor.

        Args:
            filename (:obj:`str`, optional): Path to a postprocessor that should be loaded. If not specified, a new
                postprocessor is created based on the other arguments.
            config (:obj:`dict` of :obj:`dict` of :obj:`list`of :obj:`list`): For each object name and each feature
                name, it provides a list of pair (`identifier`, `config`) that specifies a postprocessing transform.
                Postprocessing transforms are applied sequentially.
        """
        self.functions = {}
        if filename is not None:
            self.load(filename)
        else:
            self.config = kwargs.get("config", None)
            self.build_functions()

    def build_functions(self):
        """Build post-processing functions."""
        self.functions = {}
        for key, subdict in self.config.items():
            self.functions[key] = {}
            for subkey, function_configs in subdict.items():
                self.functions[key][subkey] = []
                for function_config in function_configs:
                    if len(function_config) == 1:
                        self.functions[key][subkey].append(get_transform(function_config[0]))
                    elif len(function_config) == 2:
                        self.functions[key][subkey].append(get_transform(function_config[0], **function_config[1]))
                    else:
                        raise ValueError('Too many arguments as postprocessing function config.')

    def save(self, filename):
        """Saves a postprocessor."""
        file = open(filename, 'wb')
        file.write(pickle.dumps(self.functions))
        file.close()

    def load(self, filename):
        """Loads a postprocessor."""
        file = open(filename, 'rb')
        self.functions = pickle.loads(file.read())
        file.close()

    def __call__(self, y):
        """Applies sequentially the post-processing mappings to the features contained in y."""
        return apply_postprocessing(y, self.functions)


def apply_postprocessing(y, functions):
    """Applies mapping contained in the dict `functions` to the dict `y`."""
    r = {}
    for k in y.keys():
        if k in functions.keys():
            if isinstance(y[k], dict):
                r[k] = apply_postprocessing(y[k], functions[k])
            else:
                r[k] = y[k]
                for function in functions[k]:
                    r[k] = function(r[k])
        else:
            r[k] = y[k]
    return r

