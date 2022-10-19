from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax import random
import jax.nn as jnn
from jax import vmap, jit
from functools import partial
import numpy as np
import pickle

EPS = 1e-3


def initialize_nn_params(wd, rk, scale=1e-2):
    """Initializes params of a fully connected neural network."""
    rk = random.split(rk, len(wd))
    return [initialize_layer_params(m, n, k, scale) for m, n, k in zip(wd[:-1], wd[1:], rk)]


def initialize_layer_params(m, n, key, scale=1e-2):
    """Initializes params of a neural network layer."""
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n,))


@jit
def output_nn(params, h):
    """Neural network that decodes latent variables and inputs into an output."""
    for w, b in params[:-1]:
        h = jnn.leaky_relu(jnp.dot(w, h) + b)
    final_w, final_b = params[-1]
    return jnp.dot(final_w, h) + final_b


@jit
def latent_nn(params, h):
    """Neural network that operates over latent variables, outputs values between -1 and 1."""
    for w, b in params:
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
            self.global_input_feature_names = kwargs.get('input_feglobal_input_feature_namesature_names', dict())
            self.global_output_feature_names = kwargs.get('global_output_feature_names', dict())
            self.global_object_names = list(set(list(self.global_input_feature_names.keys()) +
                                               list(self.global_output_feature_names.keys())))
            self.local_input_feature_names = kwargs.get('local_input_feature_names', dict())
            self.local_output_feature_names = kwargs.get('local_output_feature_names', dict())
            self.local_address_names = kwargs.get('local_address_names', dict())
            self.local_object_names = list(set(list(self.local_input_feature_names.keys()) +
                                               list(self.local_output_feature_names.keys()) +
                                               list(self.local_address_names.keys())))
            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.hidden_dimensions = kwargs.get('hidden_dimensions', [8])
            self.latent_dimension = kwargs.get('latent_dimension', 4)
            self.rtol = kwargs.get('rtol', 1.4e-8)
            self.atol = kwargs.get('atol', 1.4e-8)
            self.mxstep = kwargs.get('mxstep', jnp.inf)

            self.params = {}
            self.initialize_params(self.random_key)

        self.output_nn_batch = vmap(output_nn, in_axes=(None, 0), out_axes=0)
        self.latent_nn_batch = vmap(latent_nn, in_axes=(None, 0), out_axes=0)
        self.solve_and_decode_batch = vmap(self.solve_and_decode, in_axes=(None, 0))

    def save(self, filename):
        """Saves a H2MGNODE instance."""
        file = open(filename, 'wb')
        pickle.dump(self.global_input_feature_names, file)
        pickle.dump(self.global_output_feature_names, file)
        pickle.dump(self.global_object_names, file)
        pickle.dump(self.local_input_feature_names, file)
        pickle.dump(self.local_output_feature_names, file)
        pickle.dump(self.local_address_names, file)
        pickle.dump(self.local_object_names, file)
        pickle.dump(self.hidden_dimensions, file)
        pickle.dump(self.latent_dimension, file)
        pickle.dump(self.params, file)
        pickle.dump(self.rtol, file)
        pickle.dump(self.atol, file)
        pickle.dump(self.mxstep, file)
        file.close()

    def load(self, filename):
        """Reloads a H2MGNODE instance."""
        file = open(filename, 'rb')
        self.global_input_feature_names = pickle.load(file)
        self.global_output_feature_names = pickle.load(file)
        self.global_object_names = pickle.load(file)
        self.local_input_feature_names = pickle.load(file)
        self.local_output_feature_names = pickle.load(file)
        self.local_address_names = pickle.load(file)
        self.local_object_names = pickle.load(file)
        self.hidden_dimensions = pickle.load(file)
        self.latent_dimension = pickle.load(file)
        self.params = pickle.load(file)
        self.rtol = pickle.load(file)
        self.atol = pickle.load(file)
        self.mxstep = pickle.load(file)
        file.close()

    def initialize_params(self, rk):
        """Initializes all the params of the H2MGNODE instance."""
        rk_phi_c_o, rk_phi_g, rk_psi_c, rk_psi_g = random.split(rk, 4)
        self.initialize_phi_c_o_params(rk_phi_c_o)
        self.initialize_phi_g_params(rk_phi_g)
        self.initialize_psi_c_params(rk_psi_c)
        self.initialize_psi_g_params(rk_psi_g)

    def initialize_phi_c_o_params(self, rk_o):
        """Initializes params of neural network that updates latent variables of addresses."""
        self.params['phi_c_o'] = {}
        n_global_input_features = 0
        for object_name in self.global_input_feature_names.keys():
            n_global_input_features += len(self.global_input_feature_names[object_name])

        # if self.global_structure is not None:
        #     if 'input_feature_names' in list(self.global_structure.keys()):
        #         n_global_input_features = len(self.global_structure['input_feature_names'])

        rk_o = random.split(rk_o, len(self.local_object_names))
        for rk_o_k, object_name in zip(rk_o, self.local_object_names):
            self.params['phi_c_o'][object_name] = {}
            local_object_address_names = self.local_address_names.get(object_name, [])
            order = len(local_object_address_names)
            local_object_input_feature_names = self.local_input_feature_names.get(object_name, [])
            n_local_input_features = len(local_object_input_feature_names)
            # if object_name in self.local_input_feature_names.keys():
            #     n_local_input_features = len(self.local_input_feature_names[object_name])
            # if "input_feature_names" in self.local_structure[object_name].keys():
            #     n_local_input_features = len(self.local_structure[object_name]["input_feature_names"])
            rk_o_k = random.split(rk_o_k, order)
            for rk_o_k_f, address_name in zip(rk_o_k, local_object_address_names):
                nn_input_dim = n_global_input_features + n_local_input_features + (order+1) * self.latent_dimension + 1
                nn_output_dim = self.latent_dimension
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                self.params['phi_c_o'][object_name][address_name] = initialize_nn_params(wd, rk_o_k_f)

    def initialize_psi_c_params(self, rk_psi_c):
        """Initializes params of neural network that decodes latent variables into predictions."""
        self.params['psi_c'] = {}
        n_global_input_features = 0
        for object_name in self.global_input_feature_names.keys():
            n_global_input_features += len(self.global_input_feature_names[object_name])
        # n_global_input_features = 0
        # if self.global_structure is not None:
        #     if 'input_feature_names' in list(self.global_structure.keys()):
        #         n_global_input_features = len(self.global_structure['input_feature_names'])

        rk_psi_c = random.split(rk_psi_c, len(self.local_object_names))
        for rk_psi_c_k, object_name in zip(rk_psi_c, self.local_object_names):
            local_object_address_names = self.local_address_names.get(object_name, [])
            order = len(local_object_address_names)
            local_object_input_feature_names = self.local_input_feature_names.get(object_name, [])
            n_local_input_features = len(local_object_input_feature_names)
            local_object_output_feature_names = self.local_output_feature_names.get(object_name, [])
            #n_local_output_features = len(local_object_output_feature_names)

            #order = len(self.local_structure[object_name]["address_names"])

            # if "input_feature_names" in self.local_structure[object_name].keys():
            #     n_local_input_features = len(self.local_structure[object_name]["input_feature_names"])
            # else:
            #     n_local_input_features = 0

            if local_object_output_feature_names:
                self.params['psi_c'][object_name] = {}
            rk_psi_c_k = random.split(rk_psi_c_k, len(local_object_output_feature_names))
            for rk_psi_c_k_f, local_output_feature_name in zip(rk_psi_c_k, local_object_output_feature_names):
                nn_input_dim = n_global_input_features + n_local_input_features + (
                            order + 1) * self.latent_dimension + 1
                nn_output_dim = 1
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                self.params['psi_c'][object_name][local_output_feature_name] = initialize_nn_params(wd, rk_psi_c_k_f)

            # if "output_feature_names" in self.local_structure[object_name].keys():
            #     #self.params['psi_c'][object_name] = {}
            #     local_output_feature_names = self.local_structure[object_name]["output_feature_names"]
            #     rk_psi_c_k = random.split(rk_psi_c_k, len(local_output_feature_names))
            #     for rk_psi_c_k_f, local_output_feature_name in zip(rk_psi_c_k, local_output_feature_names):
            #         nn_input_dim = n_global_input_features + n_local_input_features + (order+1) * self.latent_dimension +1
            #         nn_output_dim = 1
            #         wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
            #         self.params['psi_c'][object_name][local_output_feature_name] = initialize_nn_params(wd, rk_psi_c_k_f)

    def initialize_phi_g_params(self, rk_phi_g):
        """"""
        self.params['phi_g'] = {}
        n_global_input_features = 0
        for object_name in self.global_input_feature_names.keys():
            n_global_input_features += len(self.global_input_feature_names[object_name])
        # if self.global_structure is not None:
        #     if 'input_feature_names' in list(self.global_structure.keys()):
        #         n_global_input_features = len(self.global_structure['input_feature_names'])
        #     else:
        #         n_global_input_features = 0
        nn_input_dim = n_global_input_features + 2 * self.latent_dimension + 1
        nn_output_dim = self.latent_dimension
        wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
        self.params['phi_g'] = initialize_nn_params(wd, rk_phi_g)

    def initialize_psi_g_params(self, rk_psi_g):
        """"""
        self.params['psi_g'] = {}
        n_global_input_features = 0
        for object_name in self.global_input_feature_names.keys():
            n_global_input_features += len(self.global_input_feature_names[object_name])
        # if self.global_structure is not None:
        #     if 'input_feature_names' in list(self.global_structure.keys()):
        #         n_global_input_features = len(self.global_structure['input_feature_names'])
        #     else:
        #         n_global_input_features = 0
        for object_name in self.global_output_feature_names.keys():
            self.params['psi_g'][object_name] = {}
            n_global_output_features = len(self.global_output_feature_names[object_name])
            rk_psi_g = random.split(rk_psi_g, n_global_output_features)
            for rk_psi_g_f, output_feature_name in zip(rk_psi_g, self.global_output_feature_names[object_name]):
                nn_input_dim = n_global_input_features + 2 * self.latent_dimension + 1
                nn_output_dim = 1
                wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
                self.params['psi_g'][object_name][output_feature_name] = initialize_nn_params(wd, rk_psi_g_f)

            # if 'output_feature_names' in list(self.global_structure.keys()):
            #     n_global_output_features = len(self.global_structure['output_feature_names'])
            #     rk_psi_g = random.split(rk_psi_g, n_global_output_features)
            #     for rk_psi_g_f, output_feature_name in zip(rk_psi_g, self.global_structure['output_feature_names']):
            #         nn_input_dim = n_global_input_features + 2 * self.latent_dimension + 1
            #         nn_output_dim = 1
            #         wd = [nn_input_dim, *self.hidden_dimensions, nn_output_dim]
            #         self.params['psi_g'][output_feature_name] = initialize_nn_params(wd, rk_psi_g_f)

    def forward(self, params, x):
        """Performs a forward pass for a single sample."""
        #self.check_keys(a, self.addresses), self.check_keys(x, self.input_features)
        init_state = self.init_state(x)
        return self.solve_and_decode(params, init_state)

    def forward_batch(self, params, x_batch, **kwargs):
        """Performs a forward pass for a batch of samples."""
        init_state_batch = self.init_state_batch(x_batch)
        r = self.solve_and_decode_batch(params, init_state_batch, **kwargs)
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
    def solve_and_decode(self, params, init_state, **kwargs):
        """Solves the graph dynamics and decodes to produce a meaningful output."""
        rtol = kwargs.get('rtol', self.rtol)
        atol = kwargs.get('atol', self.atol)
        mxstep = kwargs.get('mxstep', self.mxstep)
        start_and_final_state = odeint(self.dynamics, init_state, jnp.array([0., 1.]), params,
                                       rtol=rtol, atol=atol, mxstep=mxstep)
        r = self.decode_final_state(start_and_final_state, params)
        return r

    def decode_final_state(self, start_and_final_state, params):
        """Extracts the final state, and decodes it into a meaningful output."""
        fs = self.get_final_state(start_and_final_state)
        x, h_v, h_g = fs['x'], fs['h_v'], fs['h_g']
        r = {}
        for object_name in self.global_output_feature_names.keys():
            r[object_name] = {}
            for feature_name in self.global_output_feature_names[object_name]:
                w = params['psi_g'][object_name][feature_name]
                nn_input = self.get_global_nn_input(x, h_v, h_g, 0.)
                r[object_name][feature_name] = self.latent_nn_batch(w, nn_input)[:,0]
        for object_name in self.local_output_feature_names.keys():
            r[object_name] = {}
            address_names = self.local_address_names[object_name]
            for feature_name in self.local_output_feature_names[object_name]:
                w = params['psi_c'][object_name][feature_name]
                nn_input = self.get_vertex_nn_input(x, h_v, h_g, 0., object_name, address_names)
                r[object_name][feature_name] = self.latent_nn_batch(w, nn_input)[:,0]
        #
        #
        # if 'output_feature_names' in self.global_structure.keys():
        #     r['global'] = {'features': {}}
        #     for output_feature_name in self.global_structure['output_feature_names']:
        #         w = params['psi_g'][output_feature_name]
        #         nn_input = self.get_global_nn_input(x, h_v, h_g, 0.)
        #         r['global']['features'][output_feature_name] = self.latent_nn_batch(w, nn_input)[:,0]
        # for object_name in self.local_structure.keys():
        #     if object_name in x.keys():
        #         if 'output_feature_names' in self.local_structure[object_name].keys():
        #             r[object_name] = {'features': {}}
        #             for feature_name in self.local_structure[object_name]['output_feature_names']:
        #                 address_names = self.local_structure[object_name]['address_names']
        #                 w = params['psi_c'][object_name][feature_name]
        #                 nn_input = self.get_vertex_nn_input(x, h_v, h_g, 0., object_name, address_names)
        #                 r[object_name]['features'][feature_name] = self.latent_nn_batch(w, nn_input)[:,0]
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
        for object_name in self.local_address_names.keys():
            if object_name in x.keys():
                for address_name in self.local_address_names[object_name]:
                    n_obj_tot = np.maximum(n_obj_tot, np.max(x[object_name][address_name]))
        #
        # for local_feature_name in self.local_structure.keys():
        #     if local_feature_name in x.keys():
        #         address_names = self.local_structure[local_feature_name]["address_names"]
        #         for address_name in address_names:
        #             assert address_name in x[local_feature_name]["address"].keys()
        #             n_obj_tot = np.maximum(n_obj_tot, np.max(x[local_feature_name]["address"][address_name]))
        return n_obj_tot + 1

    def get_n_batch(self, x):
        """Returns the batch dimension."""
        n_batch = 0
        for object_name in x.keys():
            for feature_name in x[object_name].keys():
                n_batch = np.maximum(n_batch, np.shape(x[object_name][feature_name])[0])
        # for object_name in self.local_structure.keys():
        #     if object_name in x.keys():
        #         address_names = self.local_structure[object_name]["address_names"]
        #         for address_name in address_names:
        #             assert address_name in x[object_name]["address"].keys()
        #             n_batch = np.maximum(n_batch, np.shape(x[object_name]["address"][address_name])[0])
        return n_batch

    def dynamics(self, state, time, params):
        """Dynamics of the neural ordinary differential equation."""
        x, h_v, h_g = state['x'], state['h_v'], state['h_g']
        dx = self.constant_dynamics(x)
        dh_v = self.h_v_dynamics(x, h_v, h_g, time, params)
        dh_g = self.h_g_dynamics(x, h_v, h_g, time, params)
        return {'x': dx, 'h_v': dh_v, 'h_g': dh_g}

    def constant_dynamics(self, x):
        if isinstance(x, dict):
            r = {}
            for key, val in x.items():
                r[key] = self.constant_dynamics(val)
        else:
            r = 0.*x
        return r

    def h_v_dynamics(self, x, h_v, h_g, t, params):
        """Dynamics of the address latent variables."""
        dh_v, n = 0.*h_v, 0.*h_v + EPS
        for object_name in self.local_object_names:#self.local_structure.keys():
            if object_name in x.keys():
                #address_names = self.local_structure[object_name]["address_names"]
                address_names = self.local_address_names[object_name]

                nn_input = self.get_vertex_nn_input(x, h_v, h_g, t, object_name, address_names)

                for address_name in address_names:
                    address = x[object_name][address_name]
                    w = params['phi_c_o'][object_name][address_name]

                    update = self.latent_nn_batch(w, nn_input)
                    dh_v, n = dh_v.at[address].add(update), n.at[address].add(1 + 0. * update)
        return dh_v / n

    def get_vertex_nn_input(self, x, h_v, h_g, t, object_name, address_names):
        nn_input = []
        n_obj = jnp.shape(x[object_name][address_names[0]])[0]
        ones = jnp.ones([n_obj, 1])
        nn_input.append(h_g * ones)
        for object_name in self.global_input_feature_names.keys():
            for feature_name in self.global_input_feature_names[object_name]:
                nn_input.append(jnp.expand_dims(x[object_name][feature_name], 0))
        # if 'global' in self.local_structure.keys():
        #     if 'input_feature_names' in self.local_structure['global'].keys():
        #         for feature_name in self.local_structure['global']['input_feature_names']:
        #             nn_input.append(jnp.expand_dims(x['global'][feature_name], 0) * ones)
        for feature_name in self.local_input_feature_names[object_name]:
            nn_input.append(jnp.expand_dims(x[object_name][feature_name], 1))
        nn_input.append(t * ones)
        for address_name in address_names:
            address = x[object_name][address_name]
            nn_input.append(h_v[address])
        nn_input = jnp.concatenate(nn_input, axis=1)
        return nn_input

    def h_g_dynamics(self, x, h_v, h_g, t, params):
        """Dynamics of the global latent variable."""
        nn_input = self.get_global_nn_input(x, h_v, h_g, t)
        dh_g = self.latent_nn_batch(params['phi_g'], nn_input)
        return dh_g

    def get_global_nn_input(self, x, h_v, h_g, t):
        nn_input = [h_g]
        for object_name in self.global_input_feature_names.keys():
            for feature_name in self.global_input_feature_names[object_name]:
                nn_input.append(jnp.expand_dims(x[object_name][feature_name], 0))
        # if 'global' in self.local_structure.keys():
        #     if 'input_feature_names' in self.local_structure['global'].keys():
        #         for feature_name in self.local_structure['global']['input_feature_names']:
        #             nn_input.append(jnp.expand_dims(x['global'][feature_name], 0))
        nn_input.append(jnp.mean(h_v, axis=0, keepdims=True))
        nn_input.append(t * jnp.ones([1, 1]))
        nn_input = jnp.concatenate(nn_input, axis=1)
        return nn_input

