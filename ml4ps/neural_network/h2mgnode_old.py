
from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax import random
import jax.nn as jnn
from jax import vmap
import numpy as np
import pickle

EPS = 1e-3


def initialize_nn_weights(wd, rk):
    """Initializes weights of a fully connected neural network."""
    rk = random.split(rk, len(wd))
    return [initialize_layer_weights(m, n, k) for m, n, k in zip(wd[:-1], wd[1:], rk)]


def initialize_layer_weights(m, n, key, scale=1e-2):
    """Initializes weights of a neural network layer."""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


def output_nn(weights, h):
    """Neural network that decodes latent variables and inputs into an output."""
    for w, b in weights[:-1]:
        h = jnn.leaky_relu(jnp.dot(w, h) + b)
    final_w, final_b = weights[-1]
    return jnp.dot(final_w, h) + final_b


def latent_nn(weights, h):
    """Neural network that operates over latent variables, outputs values between -1 and 1."""
    for w, b in weights:
        h = jnn.tanh(jnp.dot(w, h) + b)
    return h


class H2MGNODE:
    """Hyper Heterogeneous Multi Graph Neural Ordinary Differential Equation



    """

    def __init__(self, file=None, **kwargs):

        if file is not None:
            self.load(file)
        else:
            try:
                self.addresses = kwargs['addresses']
                self.input_features = kwargs['input_features']
                self.output_features = kwargs['output_features']
            except:
                raise AttributeError("One should provide 'addresses', 'input_features' and 'output_features'.")

            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.time_window = kwargs.get('time_window', 1)
            self.hidden_dimensions = kwargs.get('hidden_dimensions', [8])
            self.latent_dimension = kwargs.get('latent_dimension', 4)

            self.weights = {}
            self.initialize_weights(self.random_key)

        self.output_nn_batch = vmap(output_nn, in_axes=(None, 0), out_axes=0)
        self.latent_nn_batch = vmap(latent_nn, in_axes=(None, 0), out_axes=0)
        self.solve_and_decode_batch = vmap(self.solve_and_decode, in_axes=(None, 0))

    def save(self, filename):
        """Saves a H2MGNODE instance."""
        file = open(filename, 'wb')
        pickle.dump(self.addresses, file)
        pickle.dump(self.input_features, file)
        pickle.dump(self.output_features, file)
        pickle.dump(self.time_window, file)
        pickle.dump(self.hidden_dimensions, file)
        pickle.dump(self.latent_dimension, file)
        pickle.dump(self.weights, file)
        file.close()

    def load(self, filename):
        """Reloads a H2MGNODE instance."""
        file = open(filename, 'rb')
        self.addresses = pickle.load(file)
        self.input_features = pickle.load(file)
        self.output_features = pickle.load(file)
        self.time_window = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        self.latent_dimension = pickle.load(file)
        self.weights = pickle.load(file)
        file.close()

    def initialize_weights(self, rk):
        """Initializes all the weights of the H2MGNODE instance."""
        rk_o, rk_h, rk_y = random.split(rk, 3)
        self.initialize_phi_c_o_weights(rk_o)
        self.initialize_phi_c_h_weights(rk_h)
        self.initialize_phi_c_y_weights(rk_y)

    def initialize_phi_c_o_weights(self, rk_o):
        """Initializes weights of neural network that update latent variables of addresses."""
        self.weights['phi_c_o'] = {}
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
                self.weights['phi_c_o'][k][f] = initialize_nn_weights(wd, rk_o_k_f)


    def initialize_phi_c_h_weights(self, rk_h):
        """Initializes weights of neural network that update latent variables of hyper-edges."""
        self.weights['phi_c_h'] = {}
        rk_h = random.split(rk_h, len(self.addresses.keys()))
        for rk_h_k, k in zip(rk_h, self.addresses.keys()):
            order = len(self.addresses[k])
            in_dim = len(self.input_features[k]) if k in self.input_features.keys() else 0
            nn_input_dim = (order + 1) * self.latent_dimension + in_dim * self.time_window + 1
            nn_output_dim = self.latent_dimension
            wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
            self.weights['phi_c_h'][k] = initialize_nn_weights(wd, rk_h_k)

    def initialize_phi_c_y_weights(self, rk_y):
        """Initializes weights of neural network that decode the output into a meaningful prediction."""
        self.weights['phi_c_y'] = {}
        rk_y = random.split(rk_y, len(self.output_features.keys()))
        for rk_y_k, k in zip(rk_y, self.output_features.keys()):
            self.weights['phi_c_y'][k] = {}
            rk_y_k = random.split(rk_y_k, len(k))
            for rk_y_k_f, f in zip(rk_y_k, self.output_features[k]):

                order = len(self.addresses[k])
                in_dim = len(self.input_features[k]) if k in self.input_features.keys() else 0
                nn_input_dim = (order + 1) * self.latent_dimension + in_dim * self.time_window
                nn_output_dim = self.time_window
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]

                self.weights['phi_c_y'][k][f] = initialize_nn_weights(wd, rk_y_k_f)

    def forward(self, weights, a, x):
        """Performs a forward pass for a single sample."""
        self.check_keys(a, self.addresses), self.check_keys(x, self.input_features)
        init_state = self.init_state(a, x)
        return self.solve_and_decode(weights, init_state)

    def forward_batch(self, weights, a_batch, x_batch):
        """Performs a forward pass for a batch of samples."""
        start_state_batch = self.init_state_batch(a_batch, x_batch)
        return self.solve_and_decode_batch(weights, start_state_batch)

    def init_state(self, a, x):
        """Initializes the latent state for a single sample."""
        h_v, h_e = self.init_h_v(a), self.init_h_e(a)
        return {'a': a, 'h_v': h_v, 'h_e': h_e, 'x': x}

    def init_state_batch(self, a_batch, x_batch):
        """Initializes the latent state for a batch of samples."""
        h_v_batch, h_e_batch = self.init_h_v_batch(a_batch), self.init_h_e_batch(a_batch)
        return {'a': a_batch, 'h_v': h_v_batch, 'h_e': h_e_batch, 'x': x_batch}

    def solve_and_decode(self, weights, init_state):
        """Solves the graph dynamics and decodes to produce a meaningful output."""
        start_and_final_state = odeint(self.dynamics, init_state, jnp.array([0., 1.]), weights)
        return self.decode_final_state(start_and_final_state, weights)

    def decode_final_state(self, start_and_final_state, weights):
        """Extracts the final state, and decodes it into a meaningful output."""
        fs = self.get_final_state(start_and_final_state)
        used_output = set(list(self.addresses.keys())) & set(list(fs['a'].keys())) & set(list(self.output_features.keys()))
        nn_input = self.get_nn_input(fs['a'], fs['x'], fs['h_v'], fs['h_e'])
        return {k: {f: self.output_nn_batch(weights['phi_c_y'][k][f], nn_input[k]) for f in self.output_features[k]}
                for k in used_output}

    def get_final_state(self, start_and_final_state):
        """Splits between the start and final states, and only returns the final one."""
        if isinstance(start_and_final_state, dict):
            return {k: self.get_final_state(start_and_final_state[k]) for k in start_and_final_state.keys()}
        else:
            return start_and_final_state[1]

    def init_h_v(self, a):
        """Initializes latent variables defined at addresses for a single sample."""
        n_obj_tot = self.get_n_obj_tot(a)
        return np.zeros([n_obj_tot, self.latent_dimension])

    def init_h_v_batch(self, a_batch):
        """Initializes latent variables defined at addresses for a batch of samples."""
        n_obj_tot, n_batch = self.get_n_obj_tot(a_batch), self.get_n_batch(a_batch)
        return np.zeros([n_batch, n_obj_tot, self.latent_dimension])

    def init_h_e(self, a):
        """Initializes latent variables defined at hyper-edges for a single sample."""
        n_obj = self.get_n_obj(a)
        return {k: np.zeros([n_obj_k, self.latent_dimension]) for k, n_obj_k in n_obj.items()}

    def init_h_e_batch(self, a_batch):
        """Initializes latent variables defined at hyper-edges for a batch of samples."""
        n_obj, n_batch = self.get_n_obj(a_batch), self.get_n_batch(a_batch)
        return {k: np.zeros([n_batch, n_obj_k, self.latent_dimension]) for k, n_obj_k in n_obj.items()}

    def get_n_obj_tot(self, a):
        """Returns the maximal address in the sample or batch."""
        used_addresses = set(list(self.addresses.keys())) & set(list(a.keys()))
        n_obj_tot = 0
        for k in used_addresses:
            assert set(list(self.addresses[k])).issubset(set(list(a[k].keys())))
            for f in self.addresses[k]:
                n_obj_tot = np.maximum(n_obj_tot, np.max(a[k][f]))
        return n_obj_tot + 1
        #return np.max([np.max([a_k_f for f, a_k_f in a_k.items()]) for k, a_k in a.items()]) + 1

    def get_n_obj(self, a):
        """Returns a dict of the amount of objects per class, in the sample or batch."""
        used_addresses = set(list(self.addresses.keys())) & set(list(a.keys()))
        n_obj = {}
        for k in used_addresses:
            n_obj[k] = 0
            assert set(list(self.addresses[k])).issubset(set(list(a[k].keys())))
            for f in self.addresses[k]:
                n_obj[k] = np.maximum(n_obj[k], np.shape(a[k][f])[1])
        return n_obj
        #return {k: np.max([np.shape(a_k_f)[1] for f, a_k_f in a_k.items()]) for k, a_k in a.items()}

    def get_n_batch(self, a_batch):
        """Returns the batch dimension."""
        return np.max([np.max([np.shape(a_k_f)[0] for f, a_k_f in a_k.items()]) for k, a_k in a_batch.items()])

    def dynamics(self, state, time, weights):
        """Dynamics of the neural ordinary differential equation."""
        a, x, h_v, h_e = state['a'], state['x'], state['h_v'], state['h_e']
        da, dx = self.constant_dynamics(a), self.constant_dynamics(x)
        dh_v = self.h_v_dynamics(a, x, h_v, h_e, time, weights)
        dh_e = self.h_e_dynamics(a, x, h_v, h_e, time, weights)
        return {'a': da, 'h_v': dh_v, 'h_e': dh_e, 'x': dx}

    def constant_dynamics(self, a):
        """Returns a null variation with the same structure as the input."""
        return {k: {f: 0. * a[k][f] for f in a[k].keys()} for k in a.keys()}

    def h_v_dynamics(self, a, x, h_v, h_e, t, weights):
        """Dynamics of the address latent variables."""
        dh_v, n = 0.*h_v, 0.*h_v + EPS
        used_addresses = set(list(self.addresses.keys())) & set(list(a.keys()))
        nn_input = self.get_nn_input(a, x, h_v, h_e, t)
        for k in used_addresses:
            assert set(list(self.addresses[k])).issubset(set(list(a[k].keys())))
            for f in self.addresses[k]:
                update = self.latent_nn_batch(weights['phi_c_o'][k][f], nn_input[k])
                adr = a[k][f][:, 0]
                dh_v, n = dh_v.at[adr].add(update), n.at[adr].add(1+0.*update)
        return dh_v / n

    def h_e_dynamics(self, a, x, h_v, h_e, t, weights):
        """Dynamics of the hyper-edge latent variables."""
        nn_input = self.get_nn_input(a, x, h_v, h_e, t)
        used_addresses = set(list(self.addresses.keys())) & set(list(a.keys()))
        return {k: self.latent_nn_batch(weights['phi_c_h'][k], nn_input[k]) for k in used_addresses}

    def get_nn_input(self, a, x, h_v, h_e, t=None):
        """Returns a dict of neural network inputs."""
        # TODO ici on ne veut garder que l'intersection de self.addresses.keys() et a.keys()
        used_addresses = set(list(self.addresses.keys())) & set(list(a.keys()))
        used_input_features = set(list(self.input_features.keys())) & set(list(x.keys()))
        nn_input = {k: [] for k in used_addresses}
        for k in used_addresses:

            # Get the input features x for hyper-edges of class k. There may be multiple
            # inputs, or even none.

            if k in used_input_features:
                assert set(list(self.input_features[k])).issubset(set(list(x[k].keys())))
                for f_ in self.input_features[k]:
                    nn_input[k].append(x[k][f_])

            # Get the hyper-edge latent variables for hyper-edges of class k.
            nn_input[k].append(h_e[k])

            # Get the latent variables of addresses that are used by hyper-edges of class k.
            # There are as many variables to retrieve per hyper-edge, as there are addresses
            # to which it is connected.

            assert set(list(self.addresses[k])).issubset(set(list(a[k].keys())))
            for f_ in self.addresses[k]:
                nn_input[k].append(h_v[a[k][f_][:, 0]])

            # Get also the time variable if it is given
            if t is not None:
                nn_input[k].append(t * jnp.ones([jnp.shape(h_e[k])[0], 1]))

            nn_input[k] = jnp.concatenate(nn_input[k], axis=1)
        return nn_input
