import jax.numpy as jnp
import jax.nn as jnn
import jax
from flax import linen as nn
import diffrax
from dataclasses import field
import flax
from typing import Sequence
from ml4ps.h2mg import local_addresses_iterator, H2MG

def gather_x_v(x, k):
    """Gets local input from `x` for object class `k`."""
    x_v = []
    for v in x["local_features"].get(k, {}).values():
        if len(jnp.shape(v)) == 1:
            x_v.append(jnp.expand_dims(v, axis=1))
        elif len(jnp.shape(v)) == 2:
            x_v.append(v)
        else:
            raise ValueError("Hyper-edge {} features have too many dimensions.".format(k))
    if x_v:
        return jnp.concatenate(x_v, axis=1)
    else:
        return jnp.array([[]])

def gather_h_v_local(x, h, k):
    """Gathers local hidden variables in `h` located at addresses of objects of class `k` according to `x`."""
    h_v = []
    for a in x["local_addresses"].get(k, {}).values():
        h_v.append(h["local"].at[a.astype(int)].get(mode='drop'))
    if h_v:
        return jnp.concatenate(h_v, axis=1)
    else:
        return jnp.array([[]])

def gather_h_v_mean(h):
    """Returns the mean of local hidden variables `h`."""
    if "local" in h:
        return jnp.mean(h["local"], axis=0, keepdims=True)
    else:
        return jnp.array([[]])

def gather_x_g(x):
    """Gathers global input features in `x`."""
    x_g = []
    for v in x.get("global_features", {}).values():
        if len(jnp.shape(v)) == 1:
            x_g.append(jnp.expand_dims(v, axis=1))
        elif len(jnp.shape(v)) == 2:
            x_g.append(v)
        else:
            raise ValueError("Global features have too many dimensions.")
        x_g.append(v)
    if x_g:
        return jnp.stack(x_g, axis=1)
    else:
        return jnp.array([[]])

def gather_h_g(h):
    """Gathers global latent variables in `h`."""
    if "global" in h:
        return h["global"]
    else:
        return jnp.array([[]])

def build_one_like(x, k):
    """Creates a vector filled with 1. that has the size of the amount of objects of class `k` in `x`."""
    return jnp.expand_dims(jnp.ones_like(list(x["local_addresses"][k].values())[0]), axis=1)

def gather_global_input(x, h, t):
    """Gathers input to global neural networks (both decoder and dynamics)."""
    h_v = gather_h_v_mean(h)
    h_g = gather_h_g(h)
    x_g = gather_x_g(x)
    t_vec = t * jnp.ones([1, 1])
    return jnp.nan_to_num(jnp.concatenate([h_v, h_g, x_g, t_vec], axis=1), nan=0)

def gather_local_input(x, h, t, k):
    """Gathers input to local neural networks (both decoder and dynamics)."""
    one_vec = build_one_like(x, k)
    h_v = gather_h_v_local(x, h, k)
    x_v = gather_x_v(x, k)
    h_g = gather_h_g(h) * one_vec
    x_g = gather_x_g(x) * one_vec
    t_vec = t * one_vec
    return jnp.nan_to_num(jnp.concatenate([h_v, x_v, h_g, x_g, t_vec], axis=1), nan=0)

def local_output_filter(y_local, x):
    """Reduces the output dimensionality, and replaces fictitious objects with NaN, while allowing to back-propagate."""
    for k in y_local:
        values = list(x["local_addresses"][k].values())[0]
        values = jnp.expand_dims(values, axis=1)
        for f in y_local[k]:
            y_local[k][f] = jnp.where(jnp.isnan(values+0.*y_local[k][f]), jnp.nan, y_local[k][f]) # TODO : c'est pas trivial ...
            if jnp.shape(y_local[k][f])[1] == 1:
                y_local[k][f] = y_local[k][f][:,0]
    return y_local

def global_output_filter(y_global):
    """Reduces the global output dimensionality."""
    for f in y_global:
        y_global[f] = y_global[f]
        if jnp.shape(y_global[f])[1] == 1:
            y_global[f] = y_global[f][:,0]
    return y_global


class MLP(nn.Module):
    """Multi-Layer Perceptron. Building block of the broader H2MGNODE architecture.

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of sizes of the MLP.
        out_size (:obj:`int`): Output size of the MLP.
    """

    hidden_size: Sequence[int]
    out_size: int

    @nn.compact
    def __call__(self, x):
        for i, d in enumerate(self.hidden_size):
            x = nn.Dense(d)(x)
            x = nn.tanh(x)
        return nn.Dense(self.out_size)(x)


class LocalDynamics(nn.Module):
    r"""Computes the second term of the Neural Ordinary Differential Equation that concerns local latent variables.

    .. math::
            \forall i \in \{1, \dots, n\}, \frac{dh_i}{dt} = \tanh(\sum_{(c,e,o) \in \mathcal{N}(i)} \tanh
            \Phi_\theta^{c,o} (x^g, x^c_e, h^g, (h_{o(e)})_{o\in \mathcal{O}^c}, t)

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of sizes of hidden layers of the MLPs
            :math:`(\Phi_\theta^{c,o})_{c \in \mathcal{C}, o\in \mathcal{O}^c}`.
        out_size (:obj:`int`): Output size of the MLPs
            :math:`(\Phi_\theta^{c,o})_{c \in \mathcal{C}, o\in \mathcal{O}^c}`.
    """

    hidden_size: Sequence[int]
    out_size: int

    @nn.compact
    def __call__(self, x, h, t):
        delta_sum = 0 * h['local']
        for _, k, f, a in local_addresses_iterator(x):
            delta = MLP(self.hidden_size, self.out_size, name="{}-{}".format(k, f))(gather_local_input(x, h, t, k))
            delta_sum = delta_sum.at[a.astype(int)].add(jnn.tanh(delta))
        return jnn.tanh(delta_sum)


class LocalDecoder(nn.Module):
    r"""Decodes the solution of the NODE into local predictions located at hyper-edges.

    .. math::
            \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}^c_e =
            \Psi_\theta^{c} (x^g, x^c_e, h^g(t=1), (h_{o(e)}(t=1))_{o\in \mathcal{O}^c}, t)

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): List of sizes of hidden layers of the local decoder MLPs
            :math:`(\Psi_\theta^{c})_{c \in \mathcal{C}}`.
        out_size (:obj:`int`): Output size of the local decoder MLPs
            :math:`(\Phi_\theta^{c})_{c \in \mathcal{C}}`.
        local_output_feature_names (:obj:`dict` of :obj:`float`): Provides for every object class a list of
            features for which a prediction should be output.
    """

    hidden_size: Sequence[int]
    #out_size: int
    #local_output_feature_names: dict
    local_output_feature_dimensions: dict

    @nn.compact
    def __call__(self, x, h):
        return {k: {f: MLP(self.hidden_size, dim, name="{}-{}".format(k,f))(gather_local_input(x, h, 0., k))
            for f, dim in v.items()} for k, v in self.local_output_feature_dimensions.items()}


class GlobalDynamics(nn.Module):
    r"""Computes the second term of the Neural Ordinary Differential Equation that concerns global latent variables.

    .. math::
            \frac{dh^g}{dt} = \Phi^g_\theta (x^g, h^g, \frac{1}{N}\sum_{i=1}^N h_i, t)

    Attributes:
        hidden_size (:obj:`typing.Sequence` of :obj:`int`): Sequence of sizes of hidden layers of the global dynamics
            MLPs :math:`\Phi^g_\theta`.
        out_size (:obj:`int`): Output size of the global dynamics MLPs :math:`\Phi^g_\theta`.
    """

    hidden_size: Sequence[int]
    out_size: int

    @nn.compact
    def __call__(self, x, h, t):
        return MLP(self.hidden_size, self.out_size)(gather_global_input(x, h, t))


class GlobalDecoder(nn.Module):
    r"""Decodes the solution of the NODE into a global prediction shared across the input graph.

    .. math::
        \hat{y}^g = \Psi^g_\theta (x^g, h^g(t=1), \frac{1}{N}\sum_{i=1}^N h_i(t=1), t)
    """

    hidden_size: Sequence[int]
    #out_size: int
    #global_output_feature_names: list
    local_output_feature_dimensions: dict

    @nn.compact
    def __call__(self, x, h):
        return {k: MLP(self.hidden_size, dim, name="{}".format(k))(gather_global_input(x, h, 0.))
            for k, dim in self.local_output_feature_dimensions.items()}


class H2MGNODE(flax.struct.PyTreeNode):
    r"""Hyper Heterogeneous Multi Graph Neural Ordinary Differential Equation.

    Solves the following differential system defined over a H2MG (Hyper Heterogeneous Multi Graph):

    .. math::
        \forall i \in \{1, \dots, n\}, \frac{dh_i}{dt} = \tanh(\sum_{(c,e,o) \in \mathcal{N}(i)} \tanh
            \Phi_\theta^{c,o} (x^g, x^c_e, h^g, (h_{o(e)})_{o\in \mathcal{O}^c}, t) \\
        \frac{dh^g}{dt} = \Phi^g_\theta (x^g, h^g, \frac{1}{N}\sum_{i=1}^N h_i, t) \\
        \forall c \in \mathcal{C}, \forall e \in \mathcal{E}^c, \hat{y}^c_e =
            \Psi_\theta^{c} (x^g, x^c_e, h^g(t=1), (h_{o(e)}(t=1))_{o\in \mathcal{O}^c}, t) \\
        \hat{y}^g = \Psi^g_\theta (x^g, h^g(t=1), \frac{1}{N}\sum_{i=1}^N h_i(t=1), t)

    Attributes:
        local_dynamics (:obj:`nn.Module`): Set of neural networks that define the dynamics of local hidden variables.
        global_dynamics (:obj:`nn.Module`): Neural network that define the dynamics of global hidden variables.
        local_decoder (:obj:`nn.Module`): Set of neural networks that decode hidden variables into local predictions.
        global_decoder (:obj:`nn.Module`): Set of neural networks that decode hidden variables into global predictions.
        local_latent_dimension (:obj:`int`): Dimension of local latent variables.
        global_latent_dimension (:obj:`int`): Dimension of global latent variables.
        solver (:obj:`diffrax.solver`): Neural Ordinary Differential Equation solver.
        dt0 (:obj:`int`): Neural Ordinary Differential Equation initial time step.
        stepsize_controller (:obj:`diffrax.step_size_controller`): Neural Ordinary Differential Equation step size
            controller.
        adjoint (:obj:`diffrax.adjoint`): Neural Ordinary Differential Equation adjoint method.
        max_steps (:obj:`int`): Neural Ordinary Differential Equation max amount of steps.
    """

    local_dynamics: nn.Module = None
    global_dynamics: nn.Module = None
    local_decoder: nn.Module = None
    global_decoder: nn.Module = None
    local_latent_dimension: int = 4
    global_latent_dimension: int = 4
    solver: diffrax.solver = None
    dt0: float = 0.1
    stepsize_controller: diffrax.step_size_controller = None
    adjoint: diffrax.adjoint = None
    max_steps: int = 4096

    @classmethod
    def make(cls,
             feature_dimension: H2MG,
             local_dynamics_hidden_size: Sequence[int] = None,
             global_dynamics_hidden_size: Sequence[int] = None,
             local_decoder_hidden_size: Sequence[int] = None,
             global_decoder_hidden_size: Sequence[int] = None,
             local_latent_dimension: int = 4,
             global_latent_dimension: int = 4,
             solver_name: str = "Euler",
             dt0: float = 0.1,
             stepsize_controller_name: str = "ConstantStepSize",
             stepsize_controller_kwargs: dict = None,
             adjoint_name: str = "RecursiveCheckpointAdjoint",
             max_steps: int = 4096):

        if local_dynamics_hidden_size is None:
            local_dynamics_hidden_size = [16]

        if global_dynamics_hidden_size is None:
            global_dynamics_hidden_size = [16]

        if local_decoder_hidden_size is None:
            local_decoder_hidden_size = [16]

        if global_decoder_hidden_size is None:
            global_decoder_hidden_size = [16]

        if stepsize_controller_kwargs is None:
            stepsize_controller_kwargs = dict()

        local_dynamics = LocalDynamics(local_dynamics_hidden_size, local_latent_dimension)
        global_dynamics = GlobalDynamics(global_dynamics_hidden_size, global_latent_dimension)
        local_decoder = LocalDecoder(local_decoder_hidden_size, feature_dimension.local_features)#output_feature_names["local_features"])
        global_decoder = GlobalDecoder(global_decoder_hidden_size, feature_dimension.global_features)#output_feature_names["global_features"])
        solver = eval("diffrax.{}()".format(solver_name))
        stepsize_controller = eval("diffrax.{}".format(stepsize_controller_name))(**stepsize_controller_kwargs)
        adjoint = eval("diffrax.{}()".format(adjoint_name))

        return cls(local_dynamics, global_dynamics, local_decoder, global_decoder, local_latent_dimension,
                   global_latent_dimension, solver, dt0, stepsize_controller, adjoint, max_steps)


    def init(self, rng, x):
        rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)
        h = self._initialize_latent_variables(x)
        return {
            "local_dynamics": self.local_dynamics.init(rng1, x, h, 0.),
            "global_dynamics": self.global_dynamics.init(rng2, x, h, 0.),
            "local_decoder": self.local_decoder.init(rng3, x, h),
            "global_decoder": self.global_decoder.init(rng4, x, h)
        }

    def apply(self, params, x, **kwargs):

        def f(t, y, args):
            local_delta = self.local_dynamics.apply(params["local_dynamics"], args, y, t)
            global_delta = self.global_dynamics.apply(params["global_dynamics"], args, y, t)
            return {"global": global_delta, "local": local_delta}

        term = diffrax.ODETerm(f)
        solution = diffrax.diffeqsolve(term,
                                       kwargs.get("solver", self.solver),
                                       t0=0,
                                       t1=1,
                                       dt0=kwargs.get("dt0", self.dt0),
                                       y0=self._initialize_latent_variables(x),
                                       args=x,
                                       stepsize_controller=kwargs.get("stepsize_controller", self.stepsize_controller),
                                       adjoint=kwargs.get("adjoint", self.adjoint),
                                       max_steps=kwargs.get("max_steps", self.max_steps))

        h1 = {"global": solution.ys["global"][0], "local": solution.ys["local"][0]}

        y_local = self.local_decoder.apply(params["local_decoder"], x, h1)
        y_global = self.global_decoder.apply(params["global_decoder"], x, h1)

        y_local = local_output_filter(y_local, x)
        y_global = global_output_filter(y_global)

        return H2MG({"global_features": y_global, "local_features": y_local})

    def _initialize_latent_variables(self, x):
        return {"global": jnp.zeros([1, self.global_latent_dimension]),
            "local": jnp.zeros([jnp.shape(x["all_addresses"])[0], self.local_latent_dimension])}
