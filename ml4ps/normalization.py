import numpy as np
import pickle
import tqdm

from scipy import interpolate

from ml4ps.backend.interface import collate_dict


class Normalizer:
    """Normalizes power grid features while respecting the permutation equivariance of the data.

    Attributes:
        functions (:obj:`dict` of :obj:`dict` of :obj:`ml4ps.normalization.NormalizationFunction`): Dict of dict of
            single normalizing functions. Upper level keys correspond to objects (e.g. 'load'), lower level keys
            correspond to features (e.g. 'p_mw') and the value corresponds to a normalizing function.
            Normalizing functions take scalar inputs and return scalar inputs.
    """

    def __init__(self, filename=None, **kwargs):
        """Initializes a Normalizer.

        Args:
            filename (:obj:`str`, optional): Path to a normalizer that should be loaded. If not specified,
                a new normalizer is created based on the other arguments.
            backend (:obj:`ml4ps.backend.interface.Backend`): Backend to use to extract features.
                Changing the backend will affect the objects and features names.
            data_dir (:obj:`str`): Path to the dataset that will serve to fit the normalizing functions.
            n_samples (:obj:`int`, optional): Amount of samples that should be imported from the dataset to fit the
                normalizing functions. As a matter of fact, fitting normalizing functions on a small subset of the
                dataset is faster, and usually provides a relevant normalization.
            shuffle (:obj:`bool`, optional): If true, samples used to fit the normalizing functions are drawn
                randomly from the dataset. If false, only the first samples in alphabetical order are used.
            n_breakpoints (:obj:`int`, optional): Amount of breakpoints that the piecewise linear functions should
                have. Indeed, in the case of multiple data quantiles being equal, the actual amount of breakpoints
                will be lower.
            features (:obj:`dict` of :obj:`list` of :obj:`str`): Dict of list of feature names. Keys correspond to
                objects (e.g. 'load'), and values are lists of features that should be normalized (e.g. ['p_mw',
                'q_mvar']).
        """
        self.functions = {}

        if filename is not None:
            self.load(filename)
        else:
            self.backend = kwargs.get("backend")
            self.data_dir = kwargs.get("data_dir")
            self.n_samples = kwargs.get('n_samples', 100)
            self.shuffle = kwargs.get("shuffle", False)
            self.n_breakpoints = kwargs.get('n_breakpoints', 200)
            self.feature_names = kwargs.get('feature_names', self.backend.valid_feature_names)
            self.backend.assert_names(feature_names=self.feature_names)
            #self.data_structure = kwargs.get("data_structure", self.backend.valid_data_structure)
            #self.backend.check_data_structure(self.data_structure)
            # self.features = kwargs.get("features", self.backend.valid_feature_names)
            # self.backend.check_feature_names(self.features)
            self.tqdm = kwargs.get('tqdm', tqdm.tqdm)

            self.build_functions()

    def build_functions(self):
        """Builds normalization functions.

            At first, it fetches filenames that have a valid extension, shuffling them if desired, and returning
            exactly `n_samples` of them. Then, those files are imported, and their features are extracted. Then,
            based on the obtained data, a separate normalizing function is built for each feature of each object.
        """
        data_files = self.backend.get_valid_files(self.data_dir, n_samples=self.n_samples, shuffle=self.shuffle)
        net_batch = [self.backend.load_network(file)
                     for file in self.tqdm(data_files, desc='Loading power grids.')]
        values = [self.backend.get_data_network(net, feature_names=self.feature_names)
                  for net in self.tqdm(net_batch, desc='Extracting features.')]
        values = collate_dict(values)
        self.functions = self.build_function_tree(values)

    def build_function_tree(self, values):
        r = {}
        for k in self.feature_names:
            if k in values.keys():
                r[k] = {}
                for f in self.feature_names[k]:
                    r[k][f] = NormalizationFunction(values[k][f], self.n_breakpoints)
        return r

    def build_function_tree_old(self, values):
        r = {}
        for k in values.keys():
            if isinstance(values[k], dict):
                r[k] = self.build_function_tree(values[k])
            else:
                r[k] = NormalizationFunction(values[k], self.n_breakpoints)
        return r


    def save(self, filename):
        """Saves a normalizer."""
        file = open(filename, 'wb')
        file.write(pickle.dumps(self.functions))
        file.close()

    def load(self, filename):
        """Loads a normalizer."""
        file = open(filename, 'rb')
        self.functions = pickle.loads(file.read())
        file.close()

    def __call__(self, x):
        """Normalizes input data by applying .

            **Note**
            If one feature and/or one object present in the input has no corresponding normalization function,
            then it is returned as is.
        """
        return apply_normalization(x, self.functions)


def apply_normalization(x, functions):
    r = {}
    for k in x.keys():
        if k in functions.keys():
            r[k] = {}
            for f in x[k].keys():
                if f in functions[k].keys():
                    r[k][f] = functions[k][f](x[k][f])
                else:
                    r[k][f] = x[k][f]
        else:
            r[k] = x[k]
    return r


def apply_normalization_old(x, functions):
    r = {}
    for k in x.keys():
        if k in functions.keys():
            if isinstance(x[k], dict):
                r[k] = apply_normalization(x[k], functions[k])
            else:
                r[k] = functions[k](x[k])
        else:
            r[k] = x[k]
    return r



class NormalizationFunction:
    """Normalization function that applies an approximation of the Cumulative Distribution Function.

        Attributes:
            interp_func : Piecewise linear function that will serve to normalize data.
    """

    def __init__(self, x, n_breakpoints):
        """Initializes a normalization function.

            **Note**
            In the case where all provided values are equal, there is no interpolation possible.
            Instead, the normalization function will simply subtract this unique value to its input.

            **Note**
            The piecewise linear approximation of the Cumulative Distribution Function is extended for larger (resp.
            smaller) values by extending the last (resp. first) slope.

            Args:
                x (:obj:`dict` of :obj:`dict` of :obj:`np.array`): Batch of input data which will serve to fit
                    a piecewise linear approximation of the Cumulative Distribution Function.
                n_breakpoints (:obj:`int`): Amount of breakpoints that should be present in the piecewise linear
                    approximation of the Cumulative Distribution Function.
        """
        self.p, self.q = get_proba_quantiles(x, n_breakpoints)
        self.p_merged, self.q_merged = merge_equal_quantiles(self.p, self.q)
        self.interp_func = None
        if len(self.q_merged) > 1:
            self.interp_func = interpolate.interp1d(self.q_merged, -1 + 2 * self.p_merged, fill_value="extrapolate")

    def __call__(self, x):
        """Normalizes input by applying an approximation of the CDF of values provided at initialization."""
        if self.interp_func is None:
            return x - self.q_merged
        else:
            return self.interp_func(x)


def get_proba_quantiles(x, n_breakpoints):
    """Get pairs (probability, quantile) for `n_breakpoints` equally distributed probabilities."""
    p = np.arange(0, 1, 1. / n_breakpoints)
    x_reshaped = np.reshape(x, [-1])
    x_clean = x_reshaped[~np.isnan(x_reshaped)]
    q = np.quantile(x_clean, p)
    return p, q


def merge_equal_quantiles(p, q):
    """Merges points that have the same value, by taking the mean probability."""
    q_merged, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    p_unique = 0. * q_merged
    np.add.at(p_unique, inverse, p)
    p_merged = p_unique / counts
    return p_merged, q_merged
