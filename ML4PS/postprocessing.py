import os
import pickle

import pypowsybl.network as pn
from scipy import interpolate
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp



class PostProcessor:

    def __init__(self, file=None, **kwargs):
        self.functions = {}
        if file is not None:
            self.load(file)
        else:
            self.functions = kwargs.get("functions", None)

    def save(self, filename):
        file = open(filename, 'wb')
        file.write(pickle.dumps(self.functions))
        file.close()

    def load(self, filename):
        file = open(filename, 'rb')
        self.functions = pickle.loads(file.read())
        file.close()

    def __call__(self, x):
        x_pp = {}
        for k in x.keys():
            if k in self.functions.keys():
                x_pp[k] = {}
                for f in x[k].keys():
                    if f in self.functions[k].keys():
                        x_pp[k][f] = x[k][f].copy()
                        for function in self.functions[k][f]:
                            x_pp[k][f] = function(x_pp[k][f])
                    else:
                        x_pp[k][f] = x[k][f]
            else:
                x_pp[k] = x[k]
        return x_pp


class AffineTransform:
    def __init__(self, offset=0., slope=1.):
        self.offset, self.slope = offset, slope
    def __call__(self, x):
        return self.offset + self.slope * x

class AbsValTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        return jnp.abs(x)

class TanhTransform:
    def __init__(self):
        pass
    def __call__(self, x):
        return jnp.tanh(x)