import jax.numpy as jnp


class AffineTransform:
    """Class of functions that apply an affine transform, defined by its offset and slope."""

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
    """Class of functions that return the hyperbolic tangent of the input."""

    def __init__(self):
        pass

    def __call__(self, x):
        return jnp.tanh(x)


def get_transform(identifier, **kwargs):
    """Gets the transform function associated with `identifier` with the specified keyword arguments."""
    if identifier == "affine":
        return AffineTransform(**kwargs)
    elif identifier == "abs":
        return AbsValTransform()
    elif identifier == "tanh":
        return TanhTransform()
    else:
        raise ValueError('Transform identifier {} not valid.'.format(identifier))
