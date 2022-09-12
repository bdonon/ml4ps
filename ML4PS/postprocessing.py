import pickle
import jax.numpy as jnp


class PostProcessor:
    """Savable and loadable post processor that can be applied to the output of a neural network."""

    def __init__(self, filename=None, **kwargs):
        """Initializes a PostProcessor.

        Args:
            filename (:obj:`str`, optional): Path to a postprocessor that should be loaded. If not specified, a new
                postprocessor is created based on the other arguments.
            functions (:obj:`dict` of :obj:`dict` of :obj:`list` of :obj:functions): Transforms that should be
                applied over each of the features. For each feature, multiple functions may be defined in a list,
                in which case the mappings are applied sequentially in the order of the list. Those functions should
                be objects of a class, to be savable and loadable.
        """
        self.functions = {}
        if filename is not None:
            self.load(filename)
        else:
            self.functions = kwargs.get("functions", None)

    def save(self, filename):
        """Saves a postprocessor."""
        file = open(filename, 'wb')
        file.write(pickle.dumps(self.functions))
        file.close()

    def load(self, filename):
        """Loads a postprocessor."""
        file = open(filename, 'rb')
        self.functions = pickle.loads(file.read())
        file.close()

    def __call__(self, y):
        """Applies sequentially the post-processing mappings to the features contained in y."""
        y_post = {k: {f: y[k][f] for f in y[k].keys()} for k in y.keys()}
        for k in list(set(y.keys()) & set(self.functions.keys())):
            for f in list(set(y[k].keys()) & set(self.functions[k].keys())):
                for function in self.functions[k][f]:
                    y_post[k][f] = function(y_post[k][f])
        return y_post


class AffineTransform:
    """Affine transformation, defined by its offset and slope."""

    def __init__(self, offset=0., slope=1.):
        """Initializes an affine transform (x -> offset + slope * x)."""
        self.offset, self.slope = offset, slope

    def __call__(self, x):
        return self.offset + self.slope * x


class AbsValTransform:
    """Class of functions that return the absolute value of the input."""

    def __init__(self):
        pass

    def __call__(self, x):
        return jnp.abs(x)


class TanhTransform:
    """Returns the hyperbolic tangent of the input."""

    def __init__(self):
        pass

    def __call__(self, x):
        return jnp.tanh(x)
