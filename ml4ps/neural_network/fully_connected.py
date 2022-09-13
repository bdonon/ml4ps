import jax.numpy as jnp
import jax.nn as jnn
from jax import vmap
from jax import random
import pickle
import jax


class FullyConnected:
    """Implementation of a fully connected neural network, compatible with the data"""

    def __init__(self, file=None, **kwargs):

        if file is not None:
            self.load(file)
        else:
            self.input_features = kwargs.get('input_features')
            self.output_features = kwargs.get('output_features')
            self.n_obj = kwargs.get('n_obj')
            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.hidden_dimensions = kwargs.get('hidden_dimensions', [8])

        self.input_dimension = 0
        for k in self.input_features.keys():
            n_obj_k = self.n_obj[k]
            for _ in self.input_features[k]:
                self.input_dimension += n_obj_k

        self.output_dimension = 0
        for k in self.output_features.keys():
            n_obj_k = self.n_obj[k]
            for _ in self.output_features[k]:
                self.output_dimension += n_obj_k # * self.time_window

        self.dimensions = [self.input_dimension, *self.hidden_dimensions, self.output_dimension]

        self.initialize_weights()

        self.batch_forward = vmap(self.forward_pass, in_axes=(None, 0), out_axes=0)

    def save(self, filename):
        """Saves a FC instance."""
        file = open(filename, 'wb')
        pickle.dump(self.weights, file)
        pickle.dump(self.input_features, file)
        pickle.dump(self.output_features, file)
        pickle.dump(self.n_obj, file)
        #pickle.dump(self.time_window, file)
        pickle.dump(self.hidden_dimensions, file)
        file.close()

    def load(self, filename):
        """Reloads a FC instance."""
        file = open(filename, 'rb')
        self.weights = pickle.load(file)
        self.input_features = pickle.load(file)
        self.output_features = pickle.load(file)
        self.n_obj = pickle.load(file)
        #self.time_window = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        file.close()

    def initialize_weights(self):
        keys = random.split(self.random_key, len(self.dimensions))
        def initialize_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        self.weights = [initialize_layer(m, n, k) for m, n, k in zip(self.dimensions[:-1], self.dimensions[1:], keys)]

    #def leaky_relu(self, x):
    #    return jnp.maximum(0.2 * x, x)

    def leaky_relu_layer(self, weights, x):
        return jnn.leaky_relu(jnp.dot(weights[0], x) + weights[1])

    def forward_pass(self, weights, x):
        h = self.flatten_input(x)
        #h = jax.tree_util.tree_flatten(x)
        for w, b in weights[:-1]:
            h = self.leaky_relu_layer([w, b], h)
        final_w, final_b = weights[-1]
        out = jnp.dot(final_w, h) + final_b
        out_dict = self.build_out_dict(out)
        return out_dict

    def flatten_input(self, x):
        x_flat = []
        for k in self.input_features:
            for f in self.input_features[k]:
                #n_obj = jnp.shape(x[k][f])[0]
                #ws = jnp.shape(x[k][f])[1]
                #x_flat.append(jnp.reshape(x[k][f], [n_obj * ws]))
                x_flat.append(x[k][f])
        return jnp.concatenate(x_flat)

    def build_out_dict(self, out):
        out_dict = {}
        i = 0
        for k in self.output_features.keys():
            out_dict[k] = {}
            a = self.n_obj[k]
            #b = self.time_window
            for f in self.output_features[k]:
                #out_dict[k][f] = jnp.reshape(out[i:i+a*b], [a, b])
                out_dict[k][f] = out[i:i+a]
                #i += a*b
                i += a
        return out_dict
