import numpy as np
import pickle
import tqdm

import jax.numpy as jnp

from ml4ps.h2mg import H2MG, HyperEdges, collate_h2mgs


FEATURES = "features"


def apply_function_tree(h2mg_batch, function_tree):
    h2mg_norm = H2MG()
    for k, hyper_edges in h2mg_batch.hyper_edges.items():
        if k not in function_tree:
            h2mg_norm.add_hyper_edges(k, hyper_edges)
        else:
            if (hyper_edges.features is not None) and (FEATURES in function_tree[k]):
                features_dict = {}
                for f in hyper_edges.features:
                    if f in function_tree[k][FEATURES]:
                        features_dict[f] = function_tree[k][FEATURES][f](hyper_edges.features[f])
                    else:
                        features_dict[f] = hyper_edges.features[f]
            else:
                features_dict = hyper_edges.features
            addresses_dict = hyper_edges.addresses
            h2mg_norm._add_hyper_edges(k, HyperEdges(addresses=addresses_dict, features=features_dict))
    return h2mg_norm


class H2MGNormalizer:
    """Normalizes power grid features while respecting the permutation equivariance of the data.

    Attributes:
        functions (:obj:`dict` of :obj:`dict` of :obj:`ml4ps.normalization.NormalizationFunction`): Nested dict of
            single normalizing functions. Normalizing functions take scalar inputs and return scalar inputs.
        inverse_functions (:obj:`dict` of :obj:`dict` of :obj:`ml4ps.normalization.NormalizationFunction`) : Nested
            dict of the inverse of normalizing functions.
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
        self.inverse_functions = None
        self.functions = None

        if filename is not None:
            self.load(filename)
        else:
            self.backend = kwargs.get("backend")
            self.data_dir = kwargs.get("data_dir")
            self.n_samples = kwargs.get('n_samples', 100)
            self.all_samples = kwargs.get('all_samples', True)
            self.shuffle = kwargs.get("shuffle", False)
            self.n_breakpoints = kwargs.get('n_breakpoints', 200)
            self.structure = kwargs.get('structure', self.backend.valid_structure)
            self.tqdm = kwargs.get('tqdm', tqdm.tqdm)

            self.build_functions()

    def build_functions(self):
        """Builds normalization functions.

        At first, it fetches filenames that have a valid extension, shuffling them if desired, and returning
        exactly `n_samples` of them. Then, those files are imported, and their features are extracted. Then,
        based on the obtained data, a separate normalizing function is built for each feature of each object.
        The inverse of normalizing functions is also provided.
        """
        if self.all_samples:
            data_files = self.backend.get_valid_files(self.data_dir, shuffle=self.shuffle)
        else:
            data_files = self.backend.get_valid_files(self.data_dir, n_samples=self.n_samples, shuffle=self.shuffle)
        power_grid_list = [self.backend.load_power_grid(file)
            for file in self.tqdm(data_files, desc='Building normalizer: Loading power grids.')]
        h2mg_list = [self.backend.get_h2mg_from_power_grid(power_grid, structure=self.structure)
            for power_grid in self.tqdm(power_grid_list, desc='Building normalizer: Extracting features.')]
        h2mg_batch = collate_h2mgs(h2mg_list)
        self.functions = self.build_function_tree(h2mg_batch)
        self.inverse_functions = self.build_function_tree(h2mg_batch, inverse=True)

    def build_function_tree(self, h2mg_batch, inverse=False):
        """Builds a dict structure of normalization functions, mimicking the structure of `h2mg`."""
        tree = {}
        for k, hyper_edges in h2mg_batch.hyper_edges.items():
            tree[k] = {}
            if hyper_edges.features is not None:
                tree[k][FEATURES] = {}
                for f, value in hyper_edges.features.items():
                    tree[k][FEATURES][f] = H2MGNormalizationFunction(jnp.reshape(value, [-1]), self.n_breakpoints, inverse=inverse)
        return tree
        #
        # r = h2mg * 0.#empty_like(h2mg)
        # for local_key, obj_name, feat_name, value in h2mg.local_features_iterator:
        #     r[local_key][obj_name][feat_name] = NormalizationFunction(value, self.n_breakpoints, inverse=inverse)
        # for global_key, feat_name, value in h2mg.global_features_iterator:
        #     r[global_key][feat_name] = NormalizationFunction(value, self.n_breakpoints, inverse=inverse)
        # return r

    def save(self, filename):
        """Saves a normalizer."""
        file = open(filename, 'wb')
        file.write(pickle.dumps(self.functions))
        file.write(pickle.dumps(self.inverse_functions))
        file.close()

    def load(self, filename):
        """Loads a normalizer."""
        file = open(filename, 'rb')
        self.functions = pickle.loads(file.read())
        self.inverse_functions = pickle.loads(file.read())
        file.close()

    def __call__(self, h2mg: H2MG) -> H2MG:
        """Normalizes input data."""
        return apply_function_tree(h2mg, self.functions)

    def inverse(self, h2mg: H2MG) -> H2MG:
        """De-normalizes input data by applying the inverse of normalization functions."""
        return apply_function_tree(h2mg, self.inverse_functions)


class H2MGNormalizationFunction:
    """Normalization function that applies an approximation of the Cumulative Distribution Function."""

    def __init__(self, x: jnp.array, n_breakpoints: int, inverse: bool = False):
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
        self.inverse = inverse
        self.p, self.q = get_proba_quantiles(x, n_breakpoints)
        self.p_merged, self.q_merged = merge_equal_quantiles(self.p, self.q)
        if inverse:
            self.xp, self.fp = -1 + 2 * self.p_merged, self.q_merged
        else:
            self.fp, self.xp = -1 + 2 * self.p_merged, self.q_merged

    def __call__(self, x: jnp.array) -> jnp.array:
        """Normalizes input by applying an approximation of the CDF of values provided at initialization."""
        if len(self.fp) == 1 and self.inverse:
            return x + self.fp
        elif len(self.fp) == 1 and not self.inverse:
            return x - self.xp

        else:
            interp_term = jnp.interp(x, self.xp, self.fp)
            left_term = jnp.minimum(x - self.xp[0], 0) * (self.fp[1] - self.fp[0]) / (self.xp[1] - self.xp[0])
            right_term = jnp.maximum(x - self.xp[-1], 0) * (self.fp[-1] - self.fp[-2]) / (self.xp[-1] - self.xp[-2])
            return interp_term + left_term + right_term


def get_proba_quantiles(x, n_breakpoints):
    """Get pairs (probability, quantile) for `n_breakpoints` equally distributed probabilities."""
    p = np.arange(0, 1+1./n_breakpoints, 1. / n_breakpoints)
    x_reshaped = np.reshape(x, [-1])
    x_clean = x_reshaped[~np.isnan(x_reshaped)]
    if np.any(x_clean):
        q = np.quantile(x_clean, p)
    else:
        q = 0. * p
    return p, q


def merge_equal_quantiles(p, q):
    """Merges points that have the same value, by taking the mean probability."""
    q_merged, inverse, counts = np.unique(q, return_inverse=True, return_counts=True)
    p_unique = 0. * q_merged
    np.add.at(p_unique, inverse, p)
    p_merged = p_unique / counts
    return p_merged, q_merged
