from ml4ps.utils import get_n_obj
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
            self.x = kwargs.get('x')
            self.n_obj = get_n_obj(self.x)
            self.data_structure = kwargs.get('data_structure')
            # self.input_feature_names = kwargs.get('input_feature_names')
            # self.output_feature_names = kwargs.get('output_feature_names')
            #self.n_obj = kwargs.get('n_obj')
            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.hidden_dimensions = kwargs.get('hidden_dimensions', [8])

        self.input_dimension, self.output_dimension = 0, 0
        for k in self.data_structure.keys():
            if 'input_feature_names' in self.data_structure[k].keys():
                self.input_dimension += len(self.data_structure[k]['input_feature_names']) * self.n_obj[k]
            if 'output_feature_names' in self.data_structure[k].keys():
                self.output_dimension += len(self.data_structure[k]['output_feature_names']) * self.n_obj[k]


        # self.input_dimension = 0
        # for k in self.input_feature_names.keys():
        #     n_obj_k = self.n_obj[k]
        #     for _ in self.input_feature_names[k]:
        #         self.input_dimension += n_obj_k
        #
        # self.output_dimension = 0
        # for k in self.output_feature_names.keys():
        #     n_obj_k = self.n_obj[k]
        #     for _ in self.output_feature_names[k]:
        #         self.output_dimension += n_obj_k

        self.dimensions = [self.input_dimension, *self.hidden_dimensions, self.output_dimension]

        self.initialize_weights()

        self.forward_batch = vmap(self.forward_pass, in_axes=(None, 0), out_axes=0)

    def save(self, filename):
        """Saves a FC instance."""
        file = open(filename, 'wb')
        pickle.dump(self.weights, file)
        pickle.dump(self.data_structure, file)
        # pickle.dump(self.input_feature_names, file)
        # pickle.dump(self.output_feature_names, file)
        pickle.dump(self.n_obj, file)
        #pickle.dump(self.time_window, file)
        pickle.dump(self.hidden_dimensions, file)
        file.close()

    def load(self, filename):
        """Reloads an FC instance."""
        file = open(filename, 'rb')
        self.weights = pickle.load(file)
        self.data_structure = pickle.load(file)
        # self.input_feature_names = pickle.load(file)
        # self.output_feature_names = pickle.load(file)
        self.n_obj = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        file.close()

    def initialize_weights(self):
        keys = random.split(self.random_key, len(self.dimensions))
        def initialize_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        self.weights = [initialize_layer(m, n, k) for m, n, k in zip(self.dimensions[:-1], self.dimensions[1:], keys)]

    def leaky_relu_layer(self, weights, x):
        return jnn.leaky_relu(jnp.dot(weights[0], x) + weights[1])

    def forward_pass(self, weights, x):
        h = self.flatten_input(x)
        for w, b in weights[:-1]:
            h = self.leaky_relu_layer([w, b], h)
        final_w, final_b = weights[-1]
        out = jnp.dot(final_w, h) + final_b
        out_dict = self.build_out_dict(out)
        return out_dict

    def flatten_input(self, x):
        x_flat = []
        for k in self.data_structure.keys():
            if 'input_feature_names' in self.data_structure[k].keys():
                for f in self.data_structure[k]['input_feature_names']:
                    x_flat.append(x[k]['features'][f])
        return jnp.concatenate(x_flat)

    def build_out_dict(self, out):
        r = {}
        i = 0
        for k in self.data_structure.keys():
            if 'output_feature_names' in self.data_structure[k].keys():
                a = self.n_obj[k]
                if a > 0:
                    r[k] = {'features': {}}
                    for f in self.data_structure[k]['output_feature_names']:
                        r[k]['features'][f] = out[i:i+a]
                        i += a
        return r


        # out_dict = {}
        # i = 0
        # for k in self.output_feature_names.keys():
        #     out_dict[k] = {}
        #     a = self.n_obj[k]
        #     for f in self.output_feature_names[k]:
        #         out_dict[k][f] = out[i:i+a]
        #         i += a
        # return out_dict
