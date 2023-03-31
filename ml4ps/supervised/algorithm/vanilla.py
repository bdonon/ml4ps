from abc import ABC, abstractmethod

import numpy as np
from functools import partial

from ml4ps.supervised import PSBasePb, get_model
from ml4ps.supervised.algorithm.algorithm import SupervisedAlgorithm
from ml4ps.h2mg import H2MG
from gymnasium.vector.utils.spaces import iterate
from flax.training import train_state
from tqdm import tqdm
import jax.numpy as jnp
import optax
import jax
import os
import pickle


class TrainState(train_state.TrainState):
    pass


def create_train_state(*, problem, module, apply_fn, rng, learning_rate) -> TrainState:
    """Creates an initial `TrainState`."""
    batch_x, batch_y = next(iter(problem))
    single_x = next(iterate(problem.input_space, batch_x))
    params = module.init(rng, single_x)
    tx = optax.chain(optax.clip_by_global_norm(0.1), optax.adam(learning_rate=learning_rate))
    return TrainState.create(apply_fn=apply_fn, params=params, tx=tx)


class VanillaAlgorithm(SupervisedAlgorithm):

    def __init__(self, train_problem: PSBasePb=None, validation_problem: PSBasePb=None, test_problem: PSBasePb=None, seed=0,
                 model_type: str = None, logger=None, clip_norm=0.1, learning_rate=0.0003, nn=None) -> None:
        self.train_problem = train_problem
        self.validation_problem = validation_problem
        self.test_problem = test_problem
        self.seed = seed
        self.model_type = model_type
        self.nn_kwargs = nn
        self.model = get_model(self.model_type, problem=self.train_problem, **self.nn_kwargs)
        self.logger = logger
        self.clip_norm = clip_norm
        self.learning_rate = learning_rate
        self.train_state = create_train_state(problem=self.train_problem, module=self.model, apply_fn=self.vmap_loss,
                                              rng=jax.random.PRNGKey(seed), learning_rate=learning_rate)
        super().__init__()

    @property
    def hparams(self):
        return {
            "seed": self.seed, "model_type": self.model_type, "model_kwargs": self.model_kwargs,
            "clip_norm": self.clip_norm, "learning_rate": self.learning_rate}

    @hparams.setter
    def hparams(self, value):
        self.seed = value.get("seed", self.seed)
        self.model_type = value.get("model_type", self.model_type)
        self.model_kwargs = value.get("model_kwargs", self.model_kwargs)
        self.clip_norm = value.get("clip_norm", self.clip_norm)
        self.learning_rate = value.get("learning_rate", self.learning_rate)

    def train_step(self, state: TrainState, x_batch: H2MG, y_batch: H2MG):
        loss_value, grad = self.value_and_grad_fn(state.params, x_batch, y_batch)
        state = state.apply_gradients(grads=grad)
        info = {"grad_norm": optax._src.linear_algebra.global_norm(grad), "loss_value": loss_value}
        return state, info

    def validation_step(self, state: TrainState, x_batch: H2MG, y_batch: H2MG):
        loss_value = self.vmap_loss(state.params, x_batch, y_batch)
        mse = self.vmap_metrics(state.params, x_batch, y_batch)
        return loss_value, mse

    def learn(self, logger=None, n_epochs=10):
        logger = logger or self.logger
        n_epochs = n_epochs or self.n_epochs
        step = 0
        for e in range(n_epochs):
            for x_batch, y_batch in tqdm(self.train_problem):
                self.train_state, info = self.train_step(self.train_state, x_batch, y_batch)
                logger.log_metrics_dict({"train_" + k: v for k, v in info.items()}, step)
                step += 1

            # val_metrics = {}
            # for i, x_batch, y_batch in tqdm(self.validation_problem):
            #     batch_metrics = self.validation_step(self.train_state, x_batch, y_batch)
            #     val_metrics = {k: np.concatenate([val_metrics.get(k, []), batch_metrics.get(k, [])]) for k in
            #         val_metrics | batch_metrics}
            # logger.log_metrics_dict({"val_" + k: v for k, v in val_metrics.items()}, step)

    def test(self, res_dir):
        pass
        # test_dir = os.path.join(res_dir, 'test_output')
        # if not os.path.exists(test_dir):
        #     os.mkdir(test_dir)
        # return test_model(self.test_env, self.model, self.train_state.params, output_dir=test_dir)

    @partial(jax.jit, static_argnums=(0,))
    def value_and_grad_fn(self, params, x_batch, y_batch):
        return jax.value_and_grad(self.loss_fn)(params, x_batch, y_batch)

    def loss_fn(self, params, x_batch, y_batch):
        loss_values = self.vmap_loss(params, x_batch, y_batch)
        return jnp.mean(loss_values)

    @partial(jax.jit, static_argnums=(0,))
    def vmap_loss(self, params, x_batch, y_batch):
        return jax.vmap(self.model.loss, in_axes=(None, 0, 0), out_axes=0)(params, x_batch, y_batch)

    def _model_filename(self, folder):
        return os.path.join(folder, "model.pkl")

    def _hparams_filename(self, folder):
        return os.path.join(folder, "hparams.pkl")

    def _train_state_filename(self, folder):
        return os.path.join(folder, "train_state.pkl")

    def _params_filename(self, folder):
        return os.path.join(folder, "params.pkl")

    def load(self, folder):
        with open(self._hparams_filename(folder), 'rb') as f:
            self.hparams = pickle.load(f)
        self.model = get_model(self.model_type, file=self._model_filename(folder))

    def save(self, folder):
        self.model.save(self._model_filename(folder))
        with open(self._hparams_filename(folder), 'wb') as f:
            pickle.dump(self.hparams, f)
