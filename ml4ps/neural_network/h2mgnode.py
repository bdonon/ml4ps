from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax import random
import jax.nn as jnn
from jax import vmap, jit
from functools import partial
import numpy as np
import pickle

EPS = 1e-3


def initialize_nn_weights(wd, rk, scale=1e-2):
    """Initializes weights of a fully connected neural network."""
    rk = random.split(rk, len(wd))
    return [initialize_layer_weights(m, n, k, scale) for m, n, k in zip(wd[:-1], wd[1:], rk)]


def initialize_layer_weights(m, n, key, scale=1e-2):
    """Initializes weights of a neural network layer."""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


@jit
def output_nn(weights, h):
    """Neural network that decodes latent variables and inputs into an output."""
    for w, b in weights[:-1]:
        h = jnn.leaky_relu(jnp.dot(w, h) + b)
    final_w, final_b = weights[-1]
    return jnp.dot(final_w, h) + final_b


@jit
def latent_nn(weights, h):
    """Neural network that operates over latent variables, outputs values between -1 and 1."""
    for w, b in weights:
        h = jnn.tanh(jnp.dot(w, h) + b)
    return h


def split_global_local(data_structure):
    global_structure = None
    if 'global' in data_structure.keys():
        global_structure = data_structure['global']
    local_structure = {k: v for k, v in data_structure.items() if k!='global'}
    if not local_structure:
        local_structure = None
    return global_structure, local_structure


class H2MGNODE:
    """Hyper Heterogeneous Multi Graph Neural Ordinary Differential Equation



    """

    def __init__(self, file=None, **kwargs):

        if file is not None:
            self.load(file)
        else:
            self.data_structure = kwargs.get('data_structure')
            self.global_structure, self.local_structure = split_global_local(self.data_structure)
            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
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
        pickle.dump(self.data_structure, file)
        pickle.dump(self.hidden_dimensions, file)
        pickle.dump(self.latent_dimension, file)
        pickle.dump(self.weights, file)
        file.close()

    def load(self, filename):
        """Reloads a H2MGNODE instance."""
        file = open(filename, 'rb')
        self.data_structure = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        self.latent_dimension = pickle.load(file)
        self.weights = pickle.load(file)
        file.close()

    def initialize_weights(self, rk):
        """Initializes all the weights of the H2MGNODE instance."""
        rk_phi_c_o, rk_phi_g, rk_psi_c, rk_psi_g = random.split(rk, 4)
        self.initialize_phi_c_o_weights(rk_phi_c_o)
        self.initialize_phi_g_weights(rk_phi_g)
        self.initialize_psi_c_weights(rk_psi_c)
        self.initialize_psi_g_weights(rk_psi_g)

    def initialize_phi_c_o_weights(self, rk_o):
        """Initializes weights of neural network that update latent variables of addresses."""
        self.weights['phi_c_o'] = {}
        n_global_input_features = 0
        if self.global_structure is not None:
            if 'input_feature_names' in list(self.global_structure.keys()):
                n_global_input_features = len(self.global_structure['input_feature_names'])

        rk_o = random.split(rk_o, len(self.local_structure.keys()))
        for rk_o_k, object_name in zip(rk_o, self.local_structure.keys()):
            self.weights['phi_c_o'][object_name] = {}
            order = len(self.local_structure[object_name]["address_names"])
            n_local_input_features = 0
            if "input_feature_names" in self.local_structure[object_name].keys():
                n_local_input_features = len(self.local_structure[object_name]["input_feature_names"])
            rk_o_k = random.split(rk_o_k, order)
            for rk_o_k_f, address_name in zip(rk_o_k, self.local_structure[object_name]["address_names"]):
                nn_input_dim = n_global_input_features + n_local_input_features + (order+1) * self.latent_dimension +1
                nn_output_dim = self.latent_dimension
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                self.weights['phi_c_o'][object_name][address_name] = initialize_nn_weights(wd, rk_o_k_f)

    def initialize_psi_c_weights(self, rk_psi_c):
        """Initializes weights of neural network that update latent variables of addresses."""
        self.weights['psi_c'] = {}
        n_global_input_features = 0
        if self.global_structure is not None:
            if 'input_feature_names' in list(self.global_structure.keys()):
                n_global_input_features = len(self.global_structure['input_feature_names'])

        rk_psi_c = random.split(rk_psi_c, len(self.local_structure.keys()))
        for rk_psi_c_k, object_name in zip(rk_psi_c, self.local_structure.keys()):
            order = len(self.local_structure[object_name]["address_names"])

            if "input_feature_names" in self.local_structure[object_name].keys():
                n_local_input_features = len(self.local_structure[object_name]["input_feature_names"])
            else:
                n_local_input_features = 0

            if "output_feature_names" in self.local_structure[object_name].keys():
                self.weights['psi_c'][object_name] = {}
                local_output_feature_names = self.local_structure[object_name]["output_feature_names"]
                rk_psi_c_k = random.split(rk_psi_c_k, len(local_output_feature_names))
                for rk_psi_c_k_f, local_output_feature_name in zip(rk_psi_c_k, local_output_feature_names):
                    nn_input_dim = n_global_input_features + n_local_input_features + (order+1) * self.latent_dimension +1
                    nn_output_dim = 1
                    wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                    self.weights['psi_c'][object_name][local_output_feature_name] = initialize_nn_weights(wd, rk_psi_c_k_f)

    def initialize_phi_g_weights(self, rk_phi_g):
        """"""
        self.weights['phi_g'] = {}
        if self.global_structure is not None:
            if 'input_feature_names' in list(self.global_structure.keys()):
                n_global_input_features = len(self.global_structure['input_feature_names'])
            else:
                n_global_input_features = 0
            nn_input_dim = n_global_input_features + 2 * self.latent_dimension + 1
            nn_output_dim = self.latent_dimension
            wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
            self.weights['phi_g'] = initialize_nn_weights(wd, rk_phi_g)

    def initialize_psi_g_weights(self, rk_psi_g):
        """"""
        self.weights['psi_g'] = {}
        if self.global_structure is not None:
            if 'input_feature_names' in list(self.global_structure.keys()):
                n_global_input_features = len(self.global_structure['input_feature_names'])
            else:
                n_global_input_features = 0
            if 'output_feature_names' in list(self.global_structure.keys()):
                n_global_output_features = len(self.global_structure['output_feature_names'])
                rk_psi_g = random.split(rk_psi_g, n_global_output_features)
                for rk_psi_g_f, output_feature_name in zip(rk_psi_g, self.global_structure['output_feature_names']):
                    nn_input_dim = n_global_input_features + 2 * self.latent_dimension + 1
                    nn_output_dim = 1
                    wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                    self.weights['psi_g'][output_feature_name] = initialize_nn_weights(wd, rk_psi_g_f)

    def forward(self, weights, x):
        """Performs a forward pass for a single sample."""
        #self.check_keys(a, self.addresses), self.check_keys(x, self.input_features)
        init_state = self.init_state(x)
        return self.solve_and_decode(weights, init_state)

    def forward_batch(self, weights, x_batch):
        """Performs a forward pass for a batch of samples."""
        init_state_batch = self.init_state_batch(x_batch)
        r = self.solve_and_decode_batch(weights, init_state_batch)
        return r

    def init_state(self, x):
        """Initializes the latent state for a single sample."""
        h_v, h_g = self.init_h_v(x), self.init_h_g(x)
        return {'h_v': h_v, 'h_g': h_g, 'x': x}

    def init_state_batch(self, x_batch):
        """Initializes the latent state for a batch of samples."""
        h_v_batch, h_g_batch = self.init_h_v_batch(x_batch), self.init_h_g_batch(x_batch)
        return {'h_v': h_v_batch, 'h_g': h_g_batch, 'x': x_batch}

    @partial(jit, static_argnums=(0,))
    def solve_and_decode(self, weights, init_state):
        """Solves the graph dynamics and decodes to produce a meaningful output."""
        start_and_final_state = odeint(self.dynamics, init_state, jnp.array([0., 1.]), weights)
        r = self.decode_final_state(start_and_final_state, weights)
        return r

    def decode_final_state(self, start_and_final_state, weights):
        """Extracts the final state, and decodes it into a meaningful output."""
        fs = self.get_final_state(start_and_final_state)
        x, h_v, h_g = fs['x'], fs['h_v'], fs['h_g']
        r = {}
        if 'output_feature_names' in self.global_structure.keys():
            r['global'] = {'features': {}}
            for output_feature_name in self.global_structure['output_feature_names']:
                w = weights['psi_g'][output_feature_name]
                nn_input = self.get_global_nn_input(x, h_v, h_g, 0.)
                r['global']['features'][output_feature_name] = self.latent_nn_batch(w, nn_input)[:,0]
        for object_name in self.local_structure.keys():
            if object_name in x.keys():
                if 'output_feature_names' in self.local_structure[object_name].keys():
                    r[object_name] = {'features': {}}
                    for feature_name in self.local_structure[object_name]['output_feature_names']:
                        address_names = self.local_structure[object_name]['address_names']
                        w = weights['psi_c'][object_name][feature_name]
                        nn_input = self.get_vertex_nn_input(x, h_v, h_g, 0., object_name, address_names)
                        r[object_name]['features'][feature_name] = self.latent_nn_batch(w, nn_input)[:,0]
        return r

    def get_final_state(self, start_and_final_state):
        """Splits between the start and final states, and only returns the final one."""
        if isinstance(start_and_final_state, dict):
            return {k: self.get_final_state(start_and_final_state[k]) for k in start_and_final_state.keys()}
        else:
            return start_and_final_state[1]

    def init_h_v(self, x):
        """Initializes latent variables defined at addresses for a single sample."""
        n_obj_tot = self.get_n_obj_tot(x)
        return jnp.zeros([n_obj_tot, self.latent_dimension])

    def init_h_v_batch(self, x_batch):
        """Initializes latent variables defined at addresses for a batch of samples."""
        n_obj_tot, n_batch = self.get_n_obj_tot(x_batch), self.get_n_batch(x_batch)
        return jnp.zeros([n_batch, n_obj_tot, self.latent_dimension])

    def init_h_g(self, a):
        """Initializes global latent variables shared across a single sample."""
        return jnp.zeros([1, self.latent_dimension])

    def init_h_g_batch(self, a_batch):
        """Initializes global latent variables shared accross each samples."""
        n_batch = self.get_n_batch(a_batch)
        return jnp.zeros([n_batch, 1, self.latent_dimension])

    def get_n_obj_tot(self, x):
        """Returns the maximal address in the sample or batch."""
        n_obj_tot = 0
        for local_feature_name in self.local_structure.keys():
            if local_feature_name in x.keys():
                address_names = self.local_structure[local_feature_name]["address_names"]
                for address_name in address_names:
                    assert address_name in x[local_feature_name]["address"].keys()
                    n_obj_tot = np.maximum(n_obj_tot, np.max(x[local_feature_name]["address"][address_name]))
        return n_obj_tot + 1

    def get_n_batch(self, x):
        """Returns the batch dimension."""
        n_batch = 0
        for object_name in self.local_structure.keys():
            if object_name in x.keys():
                address_names = self.local_structure[object_name]["address_names"]
                for address_name in address_names:
                    assert address_name in x[object_name]["address"].keys()
                    n_batch = np.maximum(n_batch, np.shape(x[object_name]["address"][address_name])[0])
        return n_batch

    def dynamics(self, state, time, weights):
        """Dynamics of the neural ordinary differential equation."""
        x, h_v, h_g = state['x'], state['h_v'], state['h_g']
        dx = self.constant_dynamics(x)
        dh_v = self.h_v_dynamics(x, h_v, h_g, time, weights)
        dh_g = self.h_g_dynamics(x, h_v, h_g, time, weights)
        return {'x': dx, 'h_v': dh_v, 'h_g': dh_g}

    def constant_dynamics(self, x):
        if isinstance(x, dict):
            r = {}
            for key, val in x.items():
                r[key] = self.constant_dynamics(val)
        else:
            r = 0.*x
        return r

    def h_v_dynamics(self, x, h_v, h_g, t, weights):
        """Dynamics of the address latent variables."""
        dh_v, n = 0.*h_v, 0.*h_v + EPS
        for object_name in self.local_structure.keys():
            if object_name in x.keys():
                address_names = self.local_structure[object_name]["address_names"]

                nn_input = self.get_vertex_nn_input(x, h_v, h_g, t, object_name, address_names)

                for address_name in address_names:
                    address = x[object_name]["address"][address_name]
                    w = weights['phi_c_o'][object_name][address_name]

                    update = self.latent_nn_batch(w, nn_input)
                    dh_v, n = dh_v.at[address].add(update), n.at[address].add(1 + 0. * update)
        return dh_v / n

    def get_vertex_nn_input(self, x, h_v, h_g, t, object_name, address_names):
        nn_input = []
        n_obj = jnp.shape(x[object_name]["address"][address_names[0]])[0]
        ones = jnp.ones([n_obj, 1])
        nn_input.append(h_g * ones)
        if 'global' in self.local_structure.keys():
            if 'input_feature_names' in self.local_structure['global'].keys():
                for feature_name in self.local_structure['global']['input_feature_names']:
                    nn_input.append(jnp.expand_dims(x['global'][feature_name], 0) * ones)
        for feature_name in self.local_structure[object_name]['input_feature_names']:
            nn_input.append(jnp.expand_dims(x[object_name]['features'][feature_name], 1))
        nn_input.append(t * ones)
        for address_name in address_names:
            address = x[object_name]["address"][address_name]
            nn_input.append(h_v[address])
        nn_input = jnp.concatenate(nn_input, axis=1)
        return nn_input

    def h_g_dynamics(self, x, h_v, h_g, t, weights):
        """Dynamics of the global latent variable."""
        nn_input = self.get_global_nn_input(x, h_v, h_g, t)
        dh_g = self.latent_nn_batch(weights['phi_g'], nn_input)
        return dh_g

    def get_global_nn_input(self, x, h_v, h_g, t):
        nn_input = []
        nn_input.append(h_g)
        if 'global' in self.local_structure.keys():
            if 'input_feature_names' in self.local_structure['global'].keys():
                for feature_name in self.local_structure['global']['input_feature_names']:
                    nn_input.append(jnp.expand_dims(x['global'][feature_name], 0))
        nn_input.append(jnp.mean(h_v, axis=0, keepdims=True))
        nn_input.append(t * jnp.ones([1, 1]))
        nn_input = jnp.concatenate(nn_input, axis=1)
        return nn_input

