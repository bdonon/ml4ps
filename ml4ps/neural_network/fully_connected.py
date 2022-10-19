from ml4ps.utils import get_n_obj
import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap, jit
from functools import partial
from jax import random
import pickle
import jax


class FullyConnected:
    """Implementation of a fully connected neural network, compatible with the data"""

    def __init__(self, file=None, **kwargs):

        if file is not None:
            self.load(file)
        else:
            self.x = kwargs.get('x')
            self.n_obj = get_n_obj(self.x)
            self.input_feature_names = kwargs.get('input_feature_names')
            self.output_feature_names = kwargs.get('output_feature_names')
            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.hidden_dimensions = kwargs.get('hidden_dimensions', [8])

        self.input_dimension, self.output_dimension = 0, 0
        for object_name, object_input_feature_names in self.input_feature_names.items():
            self.input_dimension += len(object_input_feature_names) * self.n_obj[object_name]
        for object_name, object_output_feature_names in self.output_feature_names.items():
            self.output_dimension += len(object_output_feature_names) * self.n_obj[object_name]

        self.dimensions = [self.input_dimension, *self.hidden_dimensions, self.output_dimension]

        self.initialize_params()

        self.forward_batch = vmap(self.forward_pass, in_axes=(None, 0), out_axes=0)

    def save(self, filename):
        """Saves a FC instance."""
        file = open(filename, 'wb')
        pickle.dump(self.params, file)
        pickle.dump(self.input_feature_names, file)
        pickle.dump(self.output_feature_names, file)
        pickle.dump(self.n_obj, file)
        pickle.dump(self.hidden_dimensions, file)
        file.close()

    def load(self, filename):
        """Reloads an FC instance."""
        file = open(filename, 'rb')
        self.params = pickle.load(file)
        # self.data_structure = pickle.load(file)
        self.input_feature_names = pickle.load(file)
        self.output_feature_names = pickle.load(file)
        self.n_obj = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        file.close()

    def initialize_params(self):
        keys = random.split(self.random_key, len(self.dimensions))
        def initialize_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        self.params = [initialize_layer(m, n, k) for m, n, k in zip(self.dimensions[:-1], self.dimensions[1:], keys)]

    def leaky_relu_layer(self, params, x):
        return jnn.leaky_relu(jnp.dot(params[0], x) + params[1])

    def forward_pass(self, params, x):
        h = self.flatten_input(x)
        for w, b in params[:-1]:
            h = self.leaky_relu_layer([w, b], h)
        final_w, final_b = params[-1]
        out = jnp.dot(final_w, h) + final_b
        out_dict = self.build_out_dict(out)
        return out_dict

    @partial(jit, static_argnums=(0,))
    def apply(self, params, x):
        return self.forward_batch(params, x)

    def flatten_input(self, x):
        x_flat = []
        for object_name, input_feature_names in self.input_feature_names.items():
            a = self.n_obj[object_name]
            if a > 0:
                for input_feature_name in input_feature_names:
                    x_flat.append(x[object_name][input_feature_name])
        return jnp.concatenate(x_flat)

    def build_out_dict(self, out):
        r = {}
        i = 0
        for object_name, output_feature_names in self.output_feature_names.items():
            a = self.n_obj[object_name]
            if a > 0:
                r[object_name] = {}
                for output_feature_name in output_feature_names:
                    r[object_name][output_feature_name] = out[i:i+a]
                    i += a
        return r
