import numpy as np
import pickle
import tqdm

from scipy import interpolate

from ML4PS.backend.interface import collate


class Normalizer:
    """Normalizes power grid features while respecting the permutation equivariance of the data.

    Attributes:
        functions (:obj:`dict` of :obj:`dict` of :obj:`ML4PS.normalization.NormalizationFunction`): Dict of dict of
            single normalizing functions. Upper level keys correspond to objects (e.g. 'load'), lower level keys
            correspond to features (e.g. 'p_mw') and the value corresponds to a normalizing function.
            Normalizing functions take scalar inputs and return scalar inputs.
    """

    def __init__(self, filename=None, **kwargs):
        """Initializes a Normalizer.

        Args:
            filename (:obj:`str`, optional): Path to a normalizer that should be loaded. If not specified, a new normalizer is
                created based on the other arguments.
            backend (:obj:`ML4PS.backend.interface.Backend`): Backend to use to extract features.
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
            self.features = kwargs.get("features", self.backend.valid_features)
            self.backend.check_features(self.features)

            self.build_functions()

    def build_functions(self):
        """Builds normalization functions.

            At first, it fetches filenames that have a valid extension, shuffling them if desired, and returning
            exactly `n_samples` of them. Then, those files are imported, and their features are extracted. Then,
            based on the obtained data, a separate normalizing function is built for each feature of each object.
        """
        print("Building a Normalizer.")
        data_files = self.backend.get_files(self.data_dir, n_samples=self.n_samples)
        network_batch = [self.backend.load_network(file) for file in tqdm.tqdm(data_files, desc='Loading power grids.')]
        values = [self.backend.extract_features(net, self.features) for net in tqdm.tqdm(network_batch,
                                                                                         desc='Extracting features.')]
        values = collate(values)
        self.functions = {k: {f: NormalizationFunction(values[k][f], self.n_breakpoints) for f in v}
                          for k, v in tqdm.tqdm(values.items(), desc='Building normalizing functions.')}
        print("Normalizer ready to normalize !")

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
        x_norm = {k: {f: x[k][f] for f in x[k].keys()} for k in x.keys()}
        for k in list(set(x.keys()) & set(self.functions.keys())):
            for f in list(set(x[k].keys()) & set(self.functions[k].keys())):
                x_norm[k][f] = self.functions[k][f](x[k][f])
        return x_norm


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
    q = np.quantile(np.reshape(x, [-1]), p)
    return p, q


def merge_equal_quantiles(p, q):
    """Merges points that have the same value, by taking the mean probability."""
    q_merged, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    p_unique = 0. * q_merged
    np.add.at(p_unique, inverse, p)
    p_merged = p_unique / counts
    return p_merged, q_merged
