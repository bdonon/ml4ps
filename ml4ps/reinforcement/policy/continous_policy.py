import pickle
from copy import deepcopy
from functools import partial
from numbers import Number
from typing import Any, Dict, Callable, Tuple

import gymnasium
import jax
import jax.numpy as jnp
import ml4ps
from gymnasium import spaces
from jax import jit, vmap
from ml4ps.h2mg import H2MG, H2MGSpace, H2MGStructure, h2mg_normal_logprob, h2mg_normal_sample, shallow_repr
from ml4ps.neural_network.base_nn import BaseNN
from ml4ps.reinforcement.policy.base import BasePolicy

def manual_normalization(_obs):  # TODO: remove this
    obs = deepcopy(_obs)
    gen_vm_pu = obs.local_hyper_edges["gen"].features["vm_pu"]
    obs.local_hyper_edges["gen"].features["vm_pu"] = (gen_vm_pu - 1) * 40 + 1 - 0.57
    ext_grid_vm_pu = obs.local_hyper_edges["ext_grid"].features["vm_pu"]
    obs.local_hyper_edges["ext_grid"].features["vm_pu"] = (ext_grid_vm_pu - 1) * 40 + 1 - 0.57
    return obs

def continuous_forward(params: Dict, observation: H2MG, normalizer: Callable, nn: BaseNN, *, mu_structure: H2MGStructure, log_sigma_structure: H2MGStructure, action_space_structure: H2MGStructure, log_sigma_0: H2MG, mu_0: H2MG,cst_sigma: float=None, clip_sigma: float=None) -> Tuple[H2MG, H2MG]:
    observation = manual_normalization(observation) # TODO: remove this
    observation = normalizer(observation)
    distrib_params = nn.apply(params, observation)
    mu, log_sigma = postprocess_continuous_distrib_params(distrib_params, mu_structure=mu_structure, log_sigma_structure=log_sigma_structure, action_space_structure=action_space_structure, log_sigma_0=log_sigma_0, mu_0=mu_0, cst_sigma=cst_sigma, clip_sigma=clip_sigma)
    return mu, log_sigma

def continuous_log_prob(params, observation, action, normalizer, nn, *, mu_structure: H2MGStructure, log_sigma_structure: H2MGStructure, action_space_structure: H2MGStructure, log_sigma_0: H2MG, mu_0: H2MG,cst_sigma: float=None, clip_sigma: float=None):
    mu_norm, log_sigma_norm = continuous_forward(params, observation, normalizer, nn, mu_structure=mu_structure, log_sigma_structure=log_sigma_structure, action_space_structure=action_space_structure, log_sigma_0=log_sigma_0, mu_0=mu_0, cst_sigma=cst_sigma, clip_sigma=clip_sigma)
    return h2mg_normal_logprob(action, mu_norm, log_sigma_norm), (mu_norm, log_sigma_norm)

def build_continuous_postprocessor(h2mg_space: H2MGSpace, mu_structure: H2MGStructure, box_to_sigma_ratio:float):
    high = h2mg_space.high
    low = h2mg_space.low
    mu_0 = H2MG.from_structure(mu_structure)
    mu_0.flat_array = (high.flat_array + low.flat_array) / 2.
    log_sigma_0 = H2MG.from_structure(mu_structure)
    log_sigma_0.flat_array = jnp.log((high.flat_array - low.flat_array) / box_to_sigma_ratio)
    return mu_0, log_sigma_0

def clip_log_sigma(log_sigma, clip_sigma, eps=0.01):
    # return jax.numpy.clip(log_sigma, a_min=jnp.log(self.clip_sigma))
    return jax.numpy.clip((1-eps)*log_sigma, a_min=jnp.log(clip_sigma)) + eps*log_sigma # soft clipping

def postprocess_continuous_distrib_params(distrib_params: H2MG,*, mu_structure: H2MGStructure, log_sigma_structure: H2MGStructure, action_space_structure: H2MGStructure, log_sigma_0: H2MG, mu_0: H2MG,cst_sigma: float=None, clip_sigma: float=None):
    mu = distrib_params.extract_from_structure(mu_structure)
    log_sigma = distrib_params.extract_from_structure(log_sigma_structure)
    mu_norm = H2MG.from_structure(action_space_structure)
    mu_norm.flat_array = jnp.exp(log_sigma_0.flat_array) * mu.flat_array + mu_0.flat_array
    log_sigma_norm = H2MG.from_structure(action_space_structure)
    log_sigma_norm.flat_array = log_sigma.flat_array + log_sigma_0.flat_array
    if clip_sigma is not None and isinstance(clip_sigma, Number):
        log_sigma_norm.flat_array = clip_log_sigma(log_sigma_norm.flat_array)
    if cst_sigma is not None and isinstance(cst_sigma, Number):
        log_sigma_norm.flat_array = jnp.full_like(log_sigma.flat_array, jnp.log(cst_sigma)) # constant sigma
    return mu_norm, log_sigma_norm

class ContinuousPolicy(BasePolicy):
    """ Continuous policy for power system control.

        Attributes
            observation_space spaces.Space
            action_space spaces.Space
            normalizer ml4ps.Normalizer preprocess input
            postprocessor postprocess output parameters
            nn produce ditribution parameters from observation input.
    """

    def __init__(self, env, normalizer=None, normalizer_args=None, nn_type="h2mgnode", box_to_sigma_ratio=8, file=None, cst_sigma=None, clip_sigma=None, nn_args={}) -> None:
        if file is not None:
            self.load(file)
        # TODO save normalizer, action, obs space, nn args
        self.box_to_sigma_ratio = box_to_sigma_ratio
        if isinstance(env, gymnasium.vector.VectorEnv):
            self.action_space = env.single_action_space.continuous
        else:
            self.action_space = env.action_space.continuous


        if normalizer is None:
            self.normalizer = self._build_normalizer(env, normalizer_args=normalizer_args)
        else:
            self.normalizer = normalizer

        self.mu_structure = self.action_space.add_suffix("_mu").structure
        self.log_sigma_structure = self.action_space.add_suffix("_log_sigma").structure

        nn_output_structure = self.mu_structure.combine(self.log_sigma_structure)
        self.nn = ml4ps.neural_network.get(nn_type, output_structure=nn_output_structure, **nn_args)
        self.mu_0, self.log_sigma_0 = self._build_postprocessor(self.action_space)

        self.cst_sigma = cst_sigma
        self.clip_sigma = clip_sigma

    def _build_postprocessor(self, h2mg_space: H2MGSpace):
        high = h2mg_space.high
        low = h2mg_space.low
        mu_0 = H2MG.from_structure(self.mu_structure)
        mu_0.flat_array = (high.flat_array + low.flat_array) / 2.
        log_sigma_0 = H2MG.from_structure(self.mu_structure)
        log_sigma_0.flat_array = jnp.log((high.flat_array - low.flat_array) / self.box_to_sigma_ratio)
        return mu_0, log_sigma_0
    
    def _clip_log_sigma(self, log_sigma, eps=0.01):
        # return jax.numpy.clip(log_sigma, a_min=jnp.log(self.clip_sigma))
        return jax.numpy.clip((1-eps)*log_sigma, a_min=jnp.log(self.clip_sigma)) + eps*log_sigma # soft clipping

    def _postprocess_distrib_params(self, distrib_params: H2MG):
        mu = distrib_params.extract_from_structure(self.mu_structure)
        log_sigma = distrib_params.extract_from_structure(self.log_sigma_structure)
        mu_norm = H2MG.from_structure(self.action_space.structure)
        mu_norm.flat_array = jnp.exp(self.log_sigma_0.flat_array) * mu.flat_array + self.mu_0.flat_array
        log_sigma_norm = H2MG.from_structure(self.action_space.structure)
        log_sigma_norm.flat_array = log_sigma.flat_array + self.log_sigma_0.flat_array
        if self.clip_sigma is not None and isinstance(self.clip_sigma, Number):
            log_sigma_norm.flat_array = self._clip_log_sigma(log_sigma_norm.flat_array)
        if self.cst_sigma is not None and isinstance(self.cst_sigma, Number):
            log_sigma_norm.flat_array = jnp.full_like(log_sigma.flat_array, jnp.log(self.cst_sigma)) # constant sigma
        # print(log_sigma_norm.flat_array))
        return mu_norm, log_sigma_norm

        # mu, log_sigma = distrib_params.extract_from_structure(self.mu_structure), distrib_params[..., 1]
        # mu_norm = self.log_sigma_0.exp() * mu + self.mu_0
        # log_sigma_norm = log_sigma + self.log_sigma_0
        # return mu_norm, log_sigma_norm

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def _check_valid_action(self, action):
        # TODO
        pass

    def log_prob(self, params, observation, action):
        mu_norm, log_sigma_norm = self.forward(params, observation)
        return h2mg_normal_logprob(action, mu_norm, log_sigma_norm), (mu_norm, log_sigma_norm)
    
    def forward(self, params, observation):
        observation = manual_normalization(observation) # TODO: remove this
        observation = self.normalizer(observation)
        distrib_params = self.nn.apply(params, observation)
        mu, log_sigma = self._postprocess_distrib_params(distrib_params)
        return mu, log_sigma
    
    def _sample(self, params, observation: H2MG, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        mu_norm, log_sigma_norm = self.forward(params, observation)
        if n_action <= 1:
            action = h2mg_normal_sample(rng, mu_norm, log_sigma_norm, deterministic=deterministic)
            log_prob = h2mg_normal_logprob(action, mu_norm, log_sigma_norm)
        else:
            action = [h2mg_normal_sample(_rng, mu_norm, log_sigma_norm, deterministic=deterministic) for _rng in jax.random.split(rng, n_action)]
            log_prob = [h2mg_normal_logprob(a, mu_norm, log_sigma_norm) for a in action]
        info = self.compute_info(mu_norm, log_sigma_norm)
        return action, log_prob, info, mu_norm, log_sigma_norm
    
    def sample(self, params, observation: H2MG, rng, deterministic=False, n_action=1):
        """Sample an action and return it together with the corresponding log probability."""
        action, log_prob, info, _, _ = self._sample(params, observation, rng, deterministic, n_action)
        return action, log_prob, info
    
    @partial(jit, static_argnums=(0, 4, 5))
    def vmap_sample(self, params, observation: spaces.Space, rng, deterministic=False, n_action=1):
        action, log_prob, _, mu, log_sigma = vmap(self._sample, in_axes=(None, 0, 0, None, None), out_axes=(0, 0, 0, 0, 0))(params, observation, rng, deterministic, n_action)
        info = self.compute_info(mu, log_sigma)
        batch_info = self.compute_batch_info(mu, log_sigma)
        return action, log_prob, info | batch_info
    
    @staticmethod
    def compute_info(mu: H2MG, log_sigma: H2MG) -> Dict[str, float]:
        mu = deepcopy(mu)
        max_mu = deepcopy(mu)
        min_mu = deepcopy(mu)
        log_sigma = deepcopy(log_sigma)
        mu.add_suffix("_mu")
        max_mu.add_suffix("_max_mu")
        min_mu.add_suffix("_min_mu")
        log_sigma.add_suffix("_log_sigma")
        info = shallow_repr(mu.apply(lambda x: jnp.asarray(jnp.nanmean(x))))
        info = info | shallow_repr(log_sigma.apply(lambda x: jnp.asarray(jnp.mean(x))))
        info = info | shallow_repr(max_mu.apply(lambda x: jnp.asarray(jnp.nanmax(x))))
        info = info | shallow_repr(min_mu.apply(lambda x: jnp.asarray(jnp.nanmin(x))))
        return info
    
    @staticmethod
    def compute_batch_info(mu: H2MG, log_sigma: H2MG) -> Dict[str, float]:
        _mu = deepcopy(mu)
        mu.add_suffix("_mu_batch_std")
        mu_std_accross_batch = shallow_repr(mu.apply(lambda x: jnp.asarray(jnp.std(x, axis=0).mean())))
        _mu.add_suffix("_mu_gen_std")
        mu_std_accross_gen = shallow_repr(_mu.apply(lambda x: jnp.asarray(jnp.std(x, axis=1).mean())))
        log_sigma.add_suffix("_log_sigma_std")
        log_sigma_std_accross_batch = shallow_repr(log_sigma.apply(lambda x: jnp.asarray(jnp.std(x, axis=0).mean())))
        return mu_std_accross_batch | log_sigma_std_accross_batch | mu_std_accross_gen

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.box_to_sigma_ratio, f)
            pickle.dump(self.action_space, f)
            pickle.dump(self.normalizer, f)
            pickle.dump(self.mu_structure, f)
            pickle.dump(self.log_sigma_structure, f)
            pickle.dump(self.nn, f)
            pickle.dump(self.mu_0, f)
            pickle.dump(self.log_sigma_0, f)
    
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.box_to_sigma_ratio = pickle.load(f)
            self.action_space = pickle.load(f)
            self.normalizer = pickle.load(f)
            self.mu_structure = pickle.load(f)
            self.log_sigma_structure = pickle.load(f)
            self.nn = pickle.load(f)
            self.mu_0 = pickle.load(f)
            self.log_sigma_0 = pickle.load(f)

    def entropy(self, log_prob_info: Dict, batch=True):
        (mu_norm, log_sigma_norm) = log_prob_info
        return jnp.zeros_like(log_sigma_norm.flat_array)
