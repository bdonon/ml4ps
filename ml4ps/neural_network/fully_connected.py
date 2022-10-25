import pickle
import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap, jit, random
from functools import partial
from ml4ps.utils import get_n_obj


def initialize_params(random_key, dimensions):
    """Initializes parameters of the neural network."""
    def initialize_layer(m, n, key, scale=1e-2):
        w_key, b_key = random.split(key)
        return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
    keys = random.split(random_key, len(dimensions))
    return [initialize_layer(m, n, k) for m, n, k in zip(dimensions[:-1], dimensions[1:], keys)]


def get_dim(feature_names, n_obj):
    """Counts the input / output dimensions based on `feature_names` and the object counts `n_obj`."""
    r = 0
    for object_name, object_input_feature_names in feature_names.items():
        r += len(object_input_feature_names) * n_obj[object_name]
    return r


def flatten_input(x, input_feature_names):
    """Flattens the H2MG input, and filters out features that are not in `input_feature_names`."""
    x_flat = []
    for object_name, object_input_feature_names in input_feature_names.items():
        if object_name in x.keys():
            for input_feature_name in object_input_feature_names:
                x_flat.append(x[object_name][input_feature_name])
    return jnp.concatenate(x_flat)


def build_out_dict(h, output_feature_names, n_obj):
    """Converts the flat output of the neural network into a nested dictionnary.

    It follows the structure defined in `output_feature_names`, and only outputs something for objects that exist
    in the input (as defined in `n_obj`).
    """
    y = {}
    i = 0
    for object_name, object_output_feature_names in output_feature_names.items():
        a = n_obj[object_name]
        if a > 0:
            y[object_name] = {}
            for output_feature_name in object_output_feature_names:
                y[object_name][output_feature_name] = h[i:i + a]
                i += a
    return y


class FullyConnected:
    """Default implementation of a Fully Connected Neural Network (MLP), compatible with H2MG data.

    Non-linearities are all Leaky-ReLU, and the last layer is indeed linear.
    """

    def __init__(self, file=None, **kwargs):
        """Initializes a Fully Connected Neural Network.

        Args:
            file (:obj:`str`): Path to a saved FullyConnected instance. If `None`, then a new model is initialized.
            x (:obj:`dict` of :obj:`dict` of :obj:`np.Array`): Batch of data. Required to define the input and output
                dimensions, and thus for the weight initialization.
            input_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that defines for each object
                class a list of feature names that should be taken as input of the neural network. Features that
                are present in the input `x` but absent from `input_feature_names` will be discarded and not passed
                to the neural network.
            output_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary that defines for each object
                class a list of feature names for which the neural network should provide a prediction. The neural
                network can only produce predictions for objects that are present in the input. If there is no
                "transformer" in the input, then the neural network will not output anything for this class.
            hidden_dim (:obj:`list` of :obj:`int`, optional): List of hidden dimensions.
            random_key (:obj:`jax.random.PRNGKey`, optional): Random key for parameters initialization
        """
        if file is not None:
            self.load(file)
        else:
            self.x = kwargs.get('x')
            self.n_obj = get_n_obj(self.x)
            self.input_feature_names = kwargs.get('input_feature_names')
            self.in_dim = get_dim(self.input_feature_names, self.n_obj)
            self.output_feature_names = kwargs.get('output_feature_names')
            self.out_dim = get_dim(self.output_feature_names, self.n_obj)
            self.hidden_dimensions = kwargs.get('hidden_dim', [8])
            self.dimensions = [self.in_dim, *self.hidden_dimensions, self.out_dim]
            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.params = initialize_params(self.random_key, self.dimensions)
        self.forward_batch = vmap(self.forward_pass, in_axes=(None, 0), out_axes=0)

    def save(self, filename):
        """Saves a FC instance."""
        file = open(filename, 'wb')
        pickle.dump(self.n_obj, file)
        pickle.dump(self.input_feature_names, file)
        pickle.dump(self.in_dim, file)
        pickle.dump(self.output_feature_names, file)
        pickle.dump(self.out_dim, file)
        pickle.dump(self.hidden_dimensions, file)
        pickle.dump(self.dimensions, file)
        pickle.dump(self.params, file)
        file.close()

    def load(self, filename):
        """Reloads an FC instance."""
        file = open(filename, 'rb')
        self.n_obj = pickle.load(file)
        self.input_feature_names = pickle.load(file)
        self.in_dim = pickle.load(file)
        self.output_feature_names = pickle.load(file)
        self.out_dim = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        self.dimensions = pickle.load(file)
        self.params = pickle.load(file)
        file.close()

    def forward_pass(self, params, x):
        """Forward pass through """
        h = flatten_input(x, self.input_feature_names)
        for w, b in params[:-1]:
            h = jnn.leaky_relu(jnp.dot(w, h) + b)
        final_w, final_b = params[-1]
        h = jnp.dot(final_w, h) + final_b
        return build_out_dict(h, self.output_feature_names, self.n_obj)

    @partial(jit, static_argnums=(0,))
    def apply(self, params, x):
        """Jitted forward pass for a batch of data."""
        return self.forward_batch(params, x)
