import jax.numpy as jnp
from jax.experimental.ode import odeint
import jax.nn as jnn
from jax import vmap
from jax import random

EPS = 1e-3


def output_fully_connected(weights, h):
    for w, b in weights[:-1]:
        h = jnn.leaky_relu(jnp.dot(w, h) + b)
    final_w, final_b = weights[-1]
    return jnp.dot(final_w, h) + final_b


def latent_fully_connected(weights, h):
    for w, b in weights[:-1]:
        h = jnn.tanh(jnp.dot(w, h) + b)
    final_w, final_b = weights[-1]
    return jnn.tanh(jnp.dot(final_w, h) + final_b)


class H2MGNODE:

    def __init__(self, options):

        self.random_key = options['random_key']
        self.addresses = options['addresses']
        self.input_features = options['input_features']
        self.output_features = options['output_features']

        # Make sure that all keys in addresses are used in either input_features or output_features or both
        # and that all features keys are associated with addresses
        assert set(list(self.input_features.keys())+list(self.output_features.keys())).issubset(self.addresses.keys())

        self.time_window = options['time_window']
        self.hidden_dimensions = options['hidden_dimensions']
        self.latent_dimension = options['latent_dimension']
        self.initialize_weights()

        self.batch_output_fully_connected = vmap(output_fully_connected, in_axes=(None, 0), out_axes=0)
        self.batch_latent_fully_connected = vmap(latent_fully_connected, in_axes=(None, 0), out_axes=0)

        self.batched_odenet = vmap(self.odenet, in_axes=(None, 0))

    def initialize_weights(self):

        def initialize_layer(m, n, key, scale=1e-2):
            w_key, b_key = random.split(key)
            return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))

        self.weights = {'phi_c_o': {}, 'phi_c_h': {}, 'phi_c_y': {}}
        rk_o, rk_h, rk_y = random.split(self.random_key, 3)

        rk_o = random.split(rk_o, len(self.addresses.keys()))
        for rk_o_k, k in zip(rk_o, self.addresses.keys()):
            self.weights['phi_c_o'][k] = {}
            rk_o_k = random.split(rk_o_k, len(k))
            for rk_o_k_f, f in zip(rk_o_k, self.addresses[k]):
                order = len(self.addresses[k])
                in_dim = len(self.input_features[k]) if k in self.input_features.keys() else 0
                nn_input_dim = (order + 1) * self.latent_dimension + in_dim * self.time_window + 1
                nn_output_dim = self.latent_dimension
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                rk_o_k_f = random.split(rk_o_k_f, len(wd))
                self.weights['phi_c_o'][k][f] = [initialize_layer(m, n, k) for m, n, k in
                                                 zip(wd[:-1], wd[1:], rk_o_k_f)]

        rk_h = random.split(rk_h, len(self.addresses.keys()))
        for rk_h_k, k in zip(rk_h, self.addresses.keys()):
            order = len(self.addresses[k])
            in_dim = len(self.input_features[k]) if k in self.input_features.keys() else 0
            nn_input_dim = (order + 1) * self.latent_dimension + (in_dim) * self.time_window + 1
            nn_output_dim = self.latent_dimension
            wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
            rk_h_k = random.split(rk_h_k, len(wd))
            self.weights['phi_c_h'][k] = [initialize_layer(m, n, k) for m, n, k in
                                          zip(wd[:-1], wd[1:], rk_h_k)]

        rk_y = random.split(rk_y, len(self.output_features.keys()))
        for rk_y_k, k in zip(rk_y, self.output_features.keys()):
            self.weights['phi_c_y'][k] = {}
            rk_y_k = random.split(rk_y_k, len(k))
            for rk_y_k_f, f in zip(rk_y_k, self.output_features[k]):
                order = len(self.addresses[k])
                in_dim = len(self.input_features[k]) if k in self.input_features.keys() else 0
                nn_input_dim = (order + 1) * self.latent_dimension + (in_dim ) * self.time_window
                nn_output_dim = self.time_window
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                rk_y_k_f = random.split(rk_y_k_f, len(wd))
                self.weights['phi_c_y'][k][f] = [initialize_layer(m, n, k) for m, n, k in
                                                 zip(wd[:-1], wd[1:], rk_y_k_f)]

    def batch_forward(self, weights, a, x):
        init_state = self.initialize_state(a, x)
        y = self.batched_odenet(weights, init_state)
        return y

    def odenet(self, weights, init_state):
        start_and_final_state = odeint(self.dynamics, init_state, jnp.array([0., 1.]), weights,
                                       rtol=1.4e-4, atol=1.4e-4, mxstep=jnp.inf)
        y = self.decode_output(start_and_final_state, weights)
        return y

    def decode_output(self, start_and_final_state, weights):
        final_state = self.get_final_state(start_and_final_state)
        a, h_v, h_e, x = final_state['a'], final_state['h_v'], final_state['h_e'], final_state['x']
        y = {}
        for k in self.output_features.keys():
            y[k] = {}
            for f in self.output_features[k]:
                neural_network_input = []
                if k in self.input_features.keys():
                    for f_ in self.input_features[k]:
                        neural_network_input.append(x[k][f_])
                neural_network_input.append(h_e[k])
                for f_ in self.addresses[k]:
                    adr = a[k][f_][:, 0]
                    neural_network_input.append(h_v[adr])
                neural_network_input = jnp.concatenate(neural_network_input, axis=1)
                y[k][f] = self.batch_output_fully_connected(weights['phi_c_y'][k][f], neural_network_input)
        return y

    def initialize_state(self, a, x, flat=False):
        h_v, h_e = self.initialize_latent_variables(a)
        if flat:
            init_state = self.flatten_batch(a, h_v, h_e, x)
        else:
            init_state = {'a': a, 'h_v': h_v, 'h_e': h_e, 'x': x}
        return init_state

    def initialize_latent_variables(self, a):
        h_v = self.initialize_h_v(a)
        h_e = self.initialize_h_e(a)
        return h_v, h_e

    def initialize_h_v(self, a):
        n_obj = 0
        n_batch = 0
        for k in self.addresses.keys():
            for f in self.addresses[k]:
                xkf = jnp.asarray(a[k][f])
                n_obj = jnp.maximum(n_obj, jnp.max(xkf) + 1)
                n_batch = jnp.maximum(n_batch, jnp.shape(xkf)[0])
        #h_v = jnp.zeros([n_obj, self.latent_dimension])
        h_v = jnp.zeros([n_batch, n_obj, self.latent_dimension])
        return h_v

    def initialize_h_e(self, a):
        h_e = {}
        n_batch = 0
        for k in self.addresses.keys():
            n_obj = 0
            for f in self.addresses[k]:
                xkf = jnp.asarray(a[k][f])
                n_obj = jnp.maximum(n_obj, xkf.shape[1])
                n_batch = jnp.maximum(n_batch, jnp.shape(xkf)[0])
            h_e[k] = jnp.zeros([n_batch, n_obj, self.latent_dimension])
        return h_e

    def dynamics(self, state, time, weights):
        a, h_v, h_e, x = state['a'], state['h_v'], state['h_e'], state['x']
        da = self.zero_update(a)
        dh_v = self.update_h_v(a, h_v, h_e, x, time, weights)
        dh_e = self.update_h_e(a, h_v, h_e, x, time, weights)
        dx = self.zero_update(x)
        return {'a': da, 'h_v': dh_v, 'h_e': dh_e, 'x': dx}

    def zero_update(self, a):
        da = {}
        for k in a.keys():
            da[k] = {}
            for f in a[k].keys():
                da[k][f] = 0. * a[k][f]
        return da

    def update_h_v(self, a, h_v, h_e, x, t, weights):
        dh_v = 0.*h_v
        n = 0.*h_v + EPS
        for k in self.addresses.keys():
            for f in self.addresses[k]:
                adr = a[k][f][:, 0]
                neural_network_input = []
                if k in self.input_features.keys():
                    for f_ in self.input_features[k]:
                        neural_network_input.append(x[k][f_])
                neural_network_input.append(h_e[k])
                for f_ in self.addresses[k]:
                    adr_ = a[k][f_][:, 0]
                    neural_network_input.append(h_v[adr_])
                neural_network_input = jnp.concatenate(neural_network_input, axis=1)
                time = t*jnp.ones([jnp.shape(neural_network_input)[0], 1])
                neural_network_input = jnp.concatenate([time, neural_network_input], axis=1)
                update = self.batch_latent_fully_connected(weights['phi_c_o'][k][f], neural_network_input)
                dh_v = dh_v.at[adr].add(update)
                n = n.at[adr].add(1+0.*update)
        return dh_v / n

    def update_h_e(self, a, h_v, h_e, x, t, weights):
        dh_e = {}
        for k in self.addresses.keys():
            neural_network_input = []
            if k in self.input_features.keys():
                for f_ in self.input_features[k]:
                    neural_network_input.append(x[k][f_])
            neural_network_input.append(h_e[k])
            for f_ in self.addresses[k]:
                adr = a[k][f_][:, 0]
                neural_network_input.append(h_v[adr])
            neural_network_input = jnp.concatenate(neural_network_input, axis=1)
            time = t*jnp.ones([jnp.shape(neural_network_input)[0], 1])
            neural_network_input = jnp.concatenate([time, neural_network_input], axis=1)
            dh_e[k] = self.batch_latent_fully_connected(weights['phi_c_h'][k], neural_network_input)
        return dh_e

    def flatten_batch(self, a, h_v, h_e, x):
        a_flat = self.flatten_batch_a(a)
        h_v_flat = self.flatten_batch_h_v(h_v)
        h_e_flat = self.flatten_batch_h_e(h_e)
        x_flat = self.flatten_batch_x(x)
        return {'a': a_flat, 'h_v': h_v_flat, 'h_e': h_e_flat, 'x': x_flat}

    def flatten_batch_a(self, a):
        a_flat = {}
        for k in self.addresses.keys():
            a_flat[k] = {}
            for f in self.addresses[k]:
                a_flat[k][f] = jnp.reshape(a[k][f], [-1, 1])
        return a_flat

    def flatten_batch_h_e(self, h_e):
        h_e_flat = {}
        for k in self.addresses.keys():
            h_e_flat[k] = jnp.reshape(h_e[k], [-1, self.latent_dimension])
        return h_e_flat

    def flatten_batch_h_v(self, h_v):
        h_v_flat = jnp.reshape(h_v, [-1, self.latent_dimension])
        return h_v_flat

    def flatten_batch_x(self, x):
        x_flat = {}
        for k in self.input_features.keys():
            x_flat[k] = {}
            for f in self.input_features[k]:
                x_flat[k][f] = jnp.reshape(x[k][f], [-1, self.time_window])
        return x_flat

    def get_final_state(self, state):
        if isinstance(state, dict):
            r = {}
            for k in state.keys():
                r[k] = self.get_final_state(state[k])
        else:
            r = state[1]
        return r

