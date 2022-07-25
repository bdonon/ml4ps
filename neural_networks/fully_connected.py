import jax.numpy as jnp
from jax.experimental.ode import odeint
import jax.nn as jnn
from jax import grad, jit, vmap
from jax import random
from functools import partial
import numpy as np

class FC:

    def __init__(self, options):

        self.key = options['key']
        self.hidden_dimensions = options['hidden_dimensions']
        self.input_features = options['input_features']
        self.output_features = options['output_features']

        self.input_dimension = 0
        for k in self.input_features.keys():
            for f in self.input_features[k].keys():
                a, b = self.input_features[k][f]
                self.input_dimension += a * b

        self.output_dimension = 0
        for k in self.output_features.keys():
            for f in self.output_features[k].keys():
                a, b = self.output_features[k][f]
                self.output_dimension += a * b

        self.dimensions = [self.input_dimension, *self.hidden_dimensions, self.output_dimension]

        self.initialize_weights()

        self.batch_forward = vmap(self.forward_pass, in_axes=(None, 0), out_axes=0)

    def initialize_weights(self):
        keys = random.split(self.key, len(self.dimensions))
        def initialize_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))
        self.weights = [initialize_layer(m, n, k) for m, n, k in zip(self.dimensions[:-1], self.dimensions[1:], keys)]

    def leaky_relu(self, x):
        return jnp.maximum(0.2 * x, x)

    def leaky_relu_layer(self, weights, x):
        return self.leaky_relu(jnp.dot(weights[0], x) + weights[1])

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
        for k in self.input_features:
            for f in self.input_features[k]:
                n_obj = jnp.shape(x[k][f])[0]
                ws = jnp.shape(x[k][f])[1]
                x_flat.append(jnp.reshape(x[k][f], [n_obj * ws]))
        return jnp.concatenate(x_flat)

    def build_out_dict(self, out):
        out_dict = {}
        i = 0
        for k in self.output_features.keys():
            out_dict[k] = {}
            for f in self.output_features[k].keys():
                a, b = self.output_features[k][f]
                out_dict[k][f] = jnp.reshape(out[i:i+a*b], [a, b])
                i += a*b
        return out_dict
