from ml4ps.supervised.algorithm.algorithm import *
from ml4ps.supervised.algorithm.vanilla import *


def get_algorithm(algorithm_type: str=None, **algorithm_kwargs):
    if algorithm_type == "vanilla":
        return VanillaAlgorithm(**algorithm_kwargs)
    else:
        raise NotImplementedError("No existing algorithm of type {}.".format(algorithm_type))
