
import ml4ps
import numpy as np
from ml4ps.supervised.model.base import BaseModel
from ml4ps.h2mg import H2MGNormalizer



class Regressor(BaseModel):
    """
    """

    def __init__(self, problem=None, normalizer_args=None, nn_type="h2mgnode", np_random=None, **nn_args) -> None:
        self.nn_args = nn_args
        self.np_random = np_random or np.random.default_rng()
        self.pb = problem
        self.input_normalizer = self._build_normalizer(problem, self.pb.input_structure, normalizer_args=normalizer_args)
        self.output_normalizer = self._build_normalizer(problem, self.pb.output_structure,
                                                        normalizer_args=normalizer_args)

        self.nn = ml4ps.neural_network.get(nn_type, output_structure=problem.output_space.continuous.structure, **nn_args)

    def init(self, rng, obs):
        return self.nn.init(rng, obs)

    def loss(self, params, x, y):
        x_norm = self.input_normalizer(x)
        y_hat_norm = self.nn.apply(params, x_norm) / 8.
        y_norm = self.output_normalizer(y)
        return ((y_hat_norm.flat_array-y_norm.flat_array)**2).nansum()

    def predict(self, params, x):
        x_norm = self.input_normalizer(x)
        y_hat_norm = self.nn.apply(params, x_norm) / 8.
        return self.output_normalizer.inverse(y_hat_norm)

    def _build_normalizer(self, pb, structure, normalizer_args=None):
        if normalizer_args is None:
            return H2MGNormalizer(backend=pb.backend, structure=structure, data_dir=pb.data_dir)
        else:
            return H2MGNormalizer(backend=pb.backend, structure=structure, data_dir=pb.data_dir, **normalizer_args)