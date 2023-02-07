from ml4ps.neural_network.fully_connected import *
from ml4ps.neural_network.h2mgnode import *


def get(identifier, config):
    if identifier == 'fully_connected':
        return FullyConnected(**config)
    elif identifier == 'h2mgnode':
        return H2MGNODE.make(**config)
    else:
        raise ValueError('Neural network identifier {} not valid.'.format(identifier))
