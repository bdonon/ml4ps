from jax.experimental.ode import odeint
import jax.numpy as jnp
from jax import random
import jax.nn as jnn
from jax import vmap, jit
from functools import partial
import numpy as np
import pickle

EPS = 1e-3


def initialize_nn_params(wd, rk, scale):
    """Initializes params of a fully connected neural network."""
    rk = random.split(rk, len(wd))
    return [initialize_layer_params(m, n, k, s) for m, n, k, s in zip(wd[:-1], wd[1:], rk, scale)]


def initialize_layer_params(m, n, key, s):
    """Initializes params of a neural network layer."""
    w_key, b_key = random.split(key)
    s_w, w_b = s[0], s[1]
    return s_w * random.normal(w_key, (n, m)), w_b * random.normal(b_key, (n,))


@jit
def output_nn(params, h):
    """Neural network that decodes latent variables and inputs into an output."""
    for w, b in params[:-1]:
        h = jnn.tanh(jnp.dot(w, h) + b)
    final_w, final_b = params[-1]
    return jnp.dot(final_w, h) + final_b


@jit
def latent_nn(params, h):
    """Neural network that operates over latent variables, outputs values between -1 and 1."""
    for w, b in params:
        h = jnn.tanh(jnp.dot(w, h) + b)
    return h


def get_n_batch(x):
    """Returns the batch dimension."""
    n_batch = 0
    for object_name in x.keys():
        for feature_name in x[object_name].keys():
            n_batch = np.maximum(n_batch, np.shape(x[object_name][feature_name])[0]).astype(int)
    return n_batch


def get_n_obj_tot(x, local_address_names):
    """Returns the maximal address in the sample or batch."""
    n_obj_tot = 0
    for object_name in local_address_names.keys():
        if object_name in x.keys():
            for address_name in local_address_names[object_name]:
                current_max = jnp.max(jnp.nan_to_num(x[object_name][address_name], nan=-1))
                n_obj_tot = np.maximum(n_obj_tot, current_max).astype(int)
    return n_obj_tot + 1


def constant_dynamics(x):
    """Returns a zero update to all components of the input `x`."""
    if isinstance(x, dict):
        return {key: constant_dynamics(val) for key, val in x.items()}
    else:
        return 0. * x


def get_global_nn_input(x, h_v, h_g, t, global_input_feature_names):
    """Returns the input for each global neural network.

    It concatenates [h_g, x_g, jnp.mean(h_v), t] into a single vector. One can observe that this vector is global.
    The local vectors `h_v` are only present in an aggregated form, by taking their mean.
    """
    nn_input = [h_g]
    for object_name in global_input_feature_names.keys():
        for feature_name in global_input_feature_names[object_name]:
            nn_input.append(jnp.expand_dims(x[object_name][feature_name], 0))
    nn_input.append(jnp.mean(h_v, axis=0, keepdims=True))
    nn_input.append(t * jnp.ones([1, 1]))
    nn_input = jnp.concatenate(nn_input, axis=1)
    return nn_input


def get_local_nn_input(x, h_v, h_g, t, global_input_feature_names, local_address_names, local_input_feature_names, replace_nan=False):
    """Returns the input for each local neural network.

    For each object class :math:`c` specified as keys of  `local_address_names`,
    and for each hyper-edge :math:`e` of such class present in `x`,
    it concatenates :math:`[h^g, x^g, x^c_e, [h^v_{o(e)}]_{o \in \mathcal{O}^c}, t]` into a single vector.

    .. note::
        The vector :math:`[h^v_{o(e)}]_{o \in \mathcal{O}^c}` is the concatenation of all address latent variables,
        to which hyper-edge :mat:`e` is connected.
        If the class of :mat:`e` is of order 1, then it contains the latent variable located at the address to which
        :mat:`e` is connected.
        If the class of :math:`e` is of order 2, then it is the concatenation of the two latent variables located
        at the two addresses to which :mat:`e` is connected. The ordering of those vectors matters : It is always the
        same.
    """
    nn_input = {}
    for object_name in local_address_names.keys():
        if object_name in x.keys():
            address_names = local_address_names[object_name]
            r = []
            n_obj = jnp.shape(x[object_name][address_names[0]])[0]
            ones = jnp.ones([n_obj, 1])
            r.append(h_g * ones)
            for global_object_name in global_input_feature_names.keys():
                for feature_name in global_input_feature_names[global_object_name]:
                    r.append(jnp.expand_dims(x[global_object_name][feature_name], 0))
            if object_name in local_input_feature_names.keys():
                for feature_name in local_input_feature_names[object_name]:
                    r.append(jnp.expand_dims(x[object_name][feature_name], 1))
            r.append(t * ones)
            for address_name in address_names:
                address = x[object_name][address_name].astype(int)
                #r.append(h_v[address])
                #r.append(h_v.at[address].get())#mode='drop'))
                r.append(h_v.at[address].get(mode='drop'))

            nn_input[object_name] = jnp.concatenate(r, axis=1)
            if replace_nan:
                nn_input[object_name] = jnp.nan_to_num(nn_input[object_name], nan=0)

    return nn_input


def get_n_global_input_features(global_input_feature_names):
    """Counts the amount of global numerical features."""
    n_global_input_features = 0
    for object_name in global_input_feature_names.keys():
        n_global_input_features += len(global_input_feature_names[object_name])
    return n_global_input_features


def get_n_local_input_features(local_input_feature_names):
    """Returns the dict that associates each object class with its amount of numerical features per object."""
    return {object_name: len(feature_names) for object_name, feature_names in local_input_feature_names.items()}


def get_n_local_addresses(local_address_names):
    """Returns the dict that associates each object class with its order (amount of ports per object)."""
    return {object_name: len(address_names) for object_name, address_names in local_address_names.items()}


def get_final_state(start_and_final_state):
    """Splits between the start and final states, and only returns the final one."""
    if isinstance(start_and_final_state, dict):
        return {k: get_final_state(start_and_final_state[k]) for k in start_and_final_state.keys()}
    else:
        return start_and_final_state[1]


def get_key_list(*args):
    """Returns the list of unique keys in a series of dictionaries."""
    return list(set([k for dict_ in args for k in dict_.keys()]))


class H2MGNODE:
    r"""Hyper Heterogeneous Multi Graph Neural Ordinary Differential Equation.

    It takes as input a Hyper Heterogeneous Multi Graph, associates each address with a latent vector,
    solves a dynamical systems where each address interact through the hyper-edges that interconnects them,
    and decodes the final state of each latent vector to produce a meaningful prediction.

    The H2MGNODE architecture relies on the following dynamical system :

    .. math::

            h_i(t=0) = [0, \dots, 0] \\
            h^g(t=0) = [0, \dots, 0] \\
            \frac{dh_i}{dt} = tanh(\sum_{(c,e,o) \in \mathcal{N}(i)} \Phi_\theta^{c,o} (x^g, x^c_e, h^g,
            (h_{o(e)})_{o\in \mathcal{O}^c}, t) \\
            \frac{dh^g}{dt} = \Phi^g_\theta (x^g, h^g, \frac{1}{N}\sum_{i=1}^N h_i, t) \\
            \hat{y}^c_e = \Psi_\theta^{c} (x^g, x^c_e, h^g(t=1), (h_{o(e)}(t=1))_{o\in \mathcal{O}^c}, t) \\
            \hat{y}^g = \Psi^g_\theta (x^g, h^g(t=1), \frac{1}{N}\sum_{i=1}^N h_i(t=1), t)


    Where :math:`\mathcal{N}(i) = \{ (c,e,o) | c \in \mathcal{C}, e \in \mathcal{E}^c, o \in \mathcal{O}^c, o(e)=i \}`.
    """

    def __init__(self, file=None, **kwargs):
        """Inits a Hyper Heterogeneous Multi Graph Neural Ordinary Differential Equation (H2MGNODE).

        Args:
            file (:obj:`str`): Path to a saved H2MGNODE instance. If `None`, then a new model is initialized.
            local_input_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary of local object
                classes and feature names that should be taken as numerical inputs.
            local_output_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary of local object
                classes and feature names for which the model should produce a numerical output.
            local_address_names (:obj:`dict` of :obj:`list` of :obj:`str`): Dictionary of local object
                classes and address names. Those address names will be extracted from the input `x` and will
                serve as interfaces between hyper-edges.
            global_input_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`, optional): Dictionary of
                global object classes and global feature names that should be taken as input of the model.
            global_output_feature_names (:obj:`dict` of :obj:`list` of :obj:`str`, optional): Dictionary of
                global object classes and global feature names that should be returned by the model.
            phi_hidden_dimensions (:obj:`list` of :obj:`int`, optional): List of hidden dimensions for the neural
                networks that define the dynamics of the system.
            psi_hidden_dimensions (:obj:`list` of :obj:`int`, optional): List of hidden dimensions for the neural
                networks that decode the final state of the dynamical system into a meaningful prediction.
            phi_scale_init (:obj:`list` of :obj:`list` of :obj:`float`, optional): List of pairs of scales that
                should be used for the initialization of dynamics neural networks. These should be consistent with
                `phi_hidden_dimensions`. For instance, if there are n hidden layers specified in
                `phi_hidden_dimensions`, then `phi_scale_init` should provide n+1 pairs of floats.
            psi_scale_init (:obj:`list` of :obj:`list` of :obj:`float`, optional): List of pairs of scales that
                should be used for the initialization of decoding neural networks. These should be consistent with
                `psi_hidden_dimensions`. For instance, if there are n hidden layers specified in
                `psi_hidden_dimensions`, then `psi_scale_init` should provide n+1 pairs of floats.
            latent_dimension (:obj:`int`, optional): Dimension of the local latent vectors located at each address,
                and of the global latent vector.
            rtol (:obj:`float`, optional): Relative local error tolerance for the Neural Ordinary Differential
                Equation solver.
            atol (:obj:`float`, optional): Absolute local error tolerance for the Neural Ordinary Differential
                Equation solver.
            mxstep (:obj:`int`, optional): Maximum number of steps to take for each time point for the Neural
                Ordinary Differential Equation solver.
            random_key (:obj:`jax.random.PRNGKey`, optional): Random key for the initialization of neural
                networks params.
        """

        if file is not None:
            self.load(file)
        else:
            self.local_input_feature_names = kwargs.get('local_input_feature_names', dict())
            self.local_output_feature_names = kwargs.get('local_output_feature_names', dict())
            self.local_address_names = kwargs.get('local_address_names', dict())
            self.local_object_names = get_key_list(self.local_input_feature_names, self.local_output_feature_names,
                                                   self.local_address_names)
            self.n_local_input_features = get_n_local_input_features(self.local_input_feature_names)
            self.n_local_addresses = get_n_local_addresses(self.local_address_names)

            self.global_input_feature_names = kwargs.get('global_input_feature_names', dict())
            self.global_output_feature_names = kwargs.get('global_output_feature_names', dict())
            self.global_object_names = get_key_list(self.global_input_feature_names, self.global_output_feature_names)
            self.n_global_input_features = get_n_global_input_features(self.global_input_feature_names)

            self.check_names()

            self.phi_hidden_dimensions = kwargs.get('phi_hidden_dimensions', [64])
            self.psi_hidden_dimensions = kwargs.get('psi_hidden_dimensions', [32, 32])
            self.phi_scale_init = kwargs.get('phi_scale_init', [[1e-1, 1e-1], [1e-1, 0]])
            self.psi_scale_init = kwargs.get('psi_scale_init', [[1e-1, 1e-1], [1e-1, 1e-1], [1e-1, 0]])
            self.latent_dimension = kwargs.get('latent_dimension', 16)

            # Neural Ordinary Differential Equation solver parameters
            self.rtol = kwargs.get('rtol', 1.4e-8)
            self.atol = kwargs.get('atol', 1.4e-8)
            self.mxstep = kwargs.get('mxstep', jnp.inf)

            self.random_key = kwargs.get('random_key', random.PRNGKey(1))
            self.params = self.initialize_params(self.random_key)

        self.output_nn_batch = vmap(output_nn, in_axes=(None, 0), out_axes=0)
        self.latent_nn_batch = vmap(latent_nn, in_axes=(None, 0), out_axes=0)
        self.solve_and_decode_batch = jit(vmap(self.solve_and_decode, in_axes=(None, 0, 0)))

    def check_names(self):
        """Checks that the various features and addresses are consistent."""
        # Make sure that all local object classes have addresses.
        assert set(self.local_object_names) == set(list(self.local_address_names.keys()))
        # Make sure that all local object classes are at least of order 1.
        for order in self.n_local_addresses.values():
            assert order > 0
        # Make sure that objects are either local or global but not both.
        assert not (set(self.local_object_names) & set(self.global_object_names))

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
        pickle.dump(self.phi_hidden_dimensions, file)
        pickle.dump(self.psi_hidden_dimensions, file)
        pickle.dump(self.phi_scale_init, file)
        pickle.dump(self.psi_scale_init, file)
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
        self.phi_hidden_dimensions = pickle.load(file)
        self.psi_hidden_dimensions = pickle.load(file)
        self.phi_scale_init = pickle.load(file)
        self.psi_scale_init = pickle.load(file)
        self.latent_dimension = pickle.load(file)
        self.params = pickle.load(file)
        self.rtol = pickle.load(file)
        self.atol = pickle.load(file)
        self.mxstep = pickle.load(file)
        file.close()

    def initialize_params(self, rk):
        """Initializes all the params of the H2MGNODE instance."""
        rk_phi_c_o, rk_phi_g, rk_psi_c, rk_psi_g = random.split(rk, 4)
        return {'phi_c': self.initialize_phi_c_params(rk_phi_c_o, self.phi_hidden_dimensions, self.phi_scale_init),
                'psi_c': self.initialize_psi_c_params(rk_psi_c, self.psi_hidden_dimensions, self.psi_scale_init),
                'phi_g': self.initialize_phi_g_params(rk_phi_g, self.phi_hidden_dimensions, self.phi_scale_init),
                'psi_g': self.initialize_psi_g_params(rk_psi_g, self.psi_hidden_dimensions, self.psi_scale_init)}

    def initialize_phi_c_params(self, rk, hidden_dimensions, scale):
        """Initializes params of the neural networks that define the dynamics of local latent variables."""
        params = {object_name: {} for object_name in self.local_address_names.keys()}
        for object_name in self.local_address_names.keys():
            nn_input_dim = self.n_global_input_features + self.n_local_input_features.get(object_name, 0) + \
                           (self.n_local_addresses[object_name] + 1) * self.latent_dimension + 1
            nn_output_dim = self.latent_dimension
            params_dim = [nn_input_dim, *hidden_dimensions, nn_output_dim]
            for address_name in self.local_address_names[object_name]:
                rk_used, rk = random.split(rk, 2)
                params[object_name][address_name] = initialize_nn_params(params_dim, rk_used, scale=scale)
        return params

    def initialize_psi_c_params(self, rk, hidden_dimensions, scale):
        """Initializes params of the neural networks that decode latent variables into local meaningful predictions."""
        params = {object_name: {} for object_name in self.local_output_feature_names.keys()}
        for object_name in self.local_output_feature_names.keys():
            nn_input_dim = self.n_global_input_features + self.n_local_input_features.get(object_name, 0) + \
                           (self.n_local_addresses[object_name] + 1) * self.latent_dimension + 1
            nn_output_dim = 1
            params_dim = [nn_input_dim, *hidden_dimensions, nn_output_dim]
            for feature_name in self.local_output_feature_names[object_name]:
                rk_used, rk = random.split(rk, 2)
                params[object_name][feature_name] = initialize_nn_params(params_dim, rk_used, scale=scale)
        return params

    def initialize_phi_g_params(self, rk, hidden_dimensions, scale):
        """Initializes params of the neural network that defines the dynamics of global latent variables."""
        nn_input_dim = self.n_global_input_features + 2 * self.latent_dimension + 1
        nn_output_dim = self.latent_dimension
        wd = [nn_input_dim, *hidden_dimensions, nn_output_dim]
        return initialize_nn_params(wd, rk, scale=scale)

    def initialize_psi_g_params(self, rk, hidden_dimensions, scale):
        """Initializes params of the neural network that decodes latent variables into global meaningful prediction."""
        params = {object_name: {} for object_name in self.global_output_feature_names.keys()}
        for object_name in self.global_output_feature_names.keys():
            for output_feature_name in self.global_output_feature_names[object_name]:
                nn_input_dim = self.n_global_input_features + 2 * self.latent_dimension + 1
                nn_output_dim = 1
                wd = [nn_input_dim, *hidden_dimensions, nn_output_dim]
                rk_used, rk = random.split(rk, 2)
                params[object_name][output_feature_name] = initialize_nn_params(wd, rk_used, scale=scale)
        return params

    def forward(self, params, x, **kwargs):
        """Performs a forward pass for a single sample."""
        init_state = self.init_state(x)
        return self.solve_and_decode(params, init_state, x, **kwargs)

    def forward_batch(self, params, x_batch, **kwargs):
        """Performs a forward pass for a batch of samples."""
        init_state_batch = self.init_state_batch(x_batch)
        return self.solve_and_decode_batch(params, init_state_batch, x_batch, **kwargs)

    def input_filter(self, x):
        """Extracts the input features from `x`.

        For the local features, it first checks that each object class keys is present in `x`.
        As a matter of fact, it is completely possible that a certain object class is present in a certain instance
        of power grid, but not in another, and this should not stop our neural network from working.
        Still, once a class of object has been identified to be present in `x`, then all the corresponding
        feature names and address names should have been provided in `x`. If one such field is missing,
        then a KeyError is returned.
        All fields that are not declared either in `self.local_input_feature_names` or in
        `self.local_address_names` but are present in `x` are simply discarded.

        For the global features, it is mandatory to provide values for all fields specified in
        `self.global_input_feature_names`. If a field is missing, then a KeyError is returned.
        All fields that are not declared in `self.global_input_feature_names` but are present in `x`
        are simply discarded.

        It also transforms nan addresses into values that will be automatically TODO
        """

        n_obj_tot = get_n_obj_tot(x, self.local_address_names)

        r = {}
        for object_name in self.local_object_names:
            if object_name in x.keys():
                r[object_name] = {}
                if object_name in self.local_input_feature_names.keys():
                    for f in self.local_input_feature_names[object_name]:
                        #r[object_name][f] = x[object_name][f]
                        r[object_name][f] = jnp.nan_to_num(x[object_name][f], nan=0.)
                if object_name in self.local_address_names.keys():
                    for f in self.local_address_names[object_name]:
                        #r[object_name][f] = x[object_name][f]
                        r[object_name][f] = jnp.nan_to_num(x[object_name][f], nan=n_obj_tot+1)
        for object_name in self.global_input_feature_names.keys():
            r[object_name] = {}
            for f in self.global_input_feature_names[object_name]:
                r[object_name][f] = x[object_name][f]
        return r

    @partial(jit, static_argnums=(0,))
    def solve_and_decode(self, params, init_state, x, **kwargs):
        """Solves the dynamics of the system and decodes the final state into a meaningful output."""
        final_state = self.solve(params, init_state, **kwargs)
        return self.decode(params, final_state, x)

    def solve(self, params, init_state, **kwargs):
        """Solves the system dynamics using the JAX implementation of Runge-Kutta, and returns the final state.

        Default solver parameters of the model can be overridden.
        """
        rtol = kwargs.get('rtol', self.rtol)
        atol = kwargs.get('atol', self.atol)
        mxstep = kwargs.get('mxstep', self.mxstep)
        trajectory = odeint(self.dynamics, init_state, jnp.array([0., 1.]), params, rtol=rtol, atol=atol, mxstep=mxstep)
        return get_final_state(trajectory)

    def decode(self, params, final_state, x):
        """Converts the final state of a latent trajectory into a meaningful prediction."""
        #x, h_v, h_g = final_state['x'], final_state['h_v'], final_state['h_g']
        x, h_v, h_g = x, final_state['h_v'], final_state['h_g']
        r_global = self.decode_global(params, x, h_v, h_g)
        r_local = self.decode_local(params, x, h_v, h_g)
        return {**r_global, **r_local}

    def decode_global(self, params, x, h_v, h_g):
        """Decodes the global information to produce global predictions."""
        r = {object_name: {} for object_name in self.global_output_feature_names.keys()}
        nn_input = get_global_nn_input(x, h_v, h_g, 0., self.global_input_feature_names)
        nn_params = params['psi_g']
        for object_name in self.global_output_feature_names.keys():
            for feature_name in self.global_output_feature_names[object_name]:
                r[object_name][feature_name] = self.output_nn_batch(nn_params[object_name][feature_name], nn_input)[:, 0]
        return r

    def decode_local(self, params, x, h_v, h_g):
        """Decodes the local information to produce hyper-edge predictions.

        TODO return nan for non existing objects

        """
        r = {}
        nn_input = get_local_nn_input(x, h_v, h_g, 0.,
                                      self.global_input_feature_names,
                                      self.local_address_names,
                                      self.local_input_feature_names,
                                      replace_nan=True)
        nn_params = params['psi_c']
        for object_name in self.local_output_feature_names.keys():
            if object_name in x.keys():
                r[object_name] = {}
                for feature_name in self.local_output_feature_names[object_name]:
                    r[object_name][feature_name] = self.output_nn_batch(nn_params[object_name][feature_name], nn_input[object_name])[:, 0]

                    for address_name in self.local_address_names[object_name]:
                        r[object_name][feature_name] = r[object_name][feature_name] + 0. * x[object_name][address_name]
        return r

    def init_state(self, x):
        """Initializes the latent state for a single sample."""
        h_v, h_g = self.init_h_v(x), self.init_h_g(x)
        return {'h_v': h_v, 'h_g': h_g, 'x': self.input_filter(x)}

    def init_state_batch(self, x_batch):
        """Initializes the latent state for a batch of samples."""
        h_v_batch, h_g_batch = self.init_h_v_batch(x_batch), self.init_h_g_batch(x_batch)
        return {'h_v': h_v_batch, 'h_g': h_g_batch, 'x': self.input_filter(x_batch)}

    def init_h_v(self, x):
        """Initializes latent variables defined at addresses for a single sample."""
        n_obj_tot = get_n_obj_tot(x, self.local_address_names)
        return jnp.zeros([n_obj_tot, self.latent_dimension])

    def init_h_v_batch(self, x_batch):
        """Initializes latent variables defined at addresses for a batch of samples."""
        n_obj_tot, n_batch = get_n_obj_tot(x_batch, self.local_address_names), get_n_batch(x_batch)
        return jnp.zeros([n_batch, n_obj_tot, self.latent_dimension])

    def init_h_g(self, a):
        """Initializes global latent variables shared across a single sample."""
        return jnp.zeros([1, self.latent_dimension])

    def init_h_g_batch(self, a_batch):
        """Initializes global latent variables shared across each sample for a batch of samples."""
        n_batch = get_n_batch(a_batch)
        return jnp.zeros([n_batch, 1, self.latent_dimension])

    def dynamics(self, state, time, params):
        """Dynamics of the neural ordinary differential equation."""
        x, h_v, h_g = state['x'], state['h_v'], state['h_g']
        dx = constant_dynamics(x)
        dh_v = self.dynamics_h_v(x, h_v, h_g, time, params)
        dh_g = self.dynamics_h_g(x, h_v, h_g, time, params)
        return {'x': dx, 'h_v': dh_v, 'h_g': dh_g}

    def dynamics_h_v(self, x, h_v, h_g, t, params):
        """Dynamics of the local address latent variables."""
        dh_v = 0. * h_v
        nn_input = get_local_nn_input(x, h_v, h_g, t,
                                      self.global_input_feature_names,
                                      self.local_address_names,
                                      self.local_input_feature_names,
                                      replace_nan=True)
        nn_params = params['phi_c']
        for object_name in self.local_address_names.keys():
            if object_name in x.keys():
                for address_name in self.local_address_names[object_name]:
                    address_value = x[object_name][address_name].astype(int)
                    update = self.latent_nn_batch(nn_params[object_name][address_name], nn_input[object_name])
                    dh_v = dh_v.at[address_value].add(update)#, mode="drop")

        return jnn.tanh(dh_v)

    def dynamics_h_g(self, x, h_v, h_g, t, params):
        """Dynamics of the global latent variable."""
        nn_input = get_global_nn_input(x, h_v, h_g, t, self.global_input_feature_names)
        nn_params = params['phi_g']
        return self.latent_nn_batch(nn_params, nn_input)

    def apply(self, params, x):
        """Forward pass for a batch of data."""
        return self.forward_batch(params, x)

