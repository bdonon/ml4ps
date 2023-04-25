from ml4ps.reinforcement.algorithm.reinforce import Reinforce
from ml4ps.reinforcement.algorithm.reinforce_nn_baseline import ReinforceBaseline
from ml4ps.reinforcement.algorithm.algorithm import Algorithm

def get_algorithm(algorithm_type, *args, **kwargs) -> Algorithm:
    if algorithm_type =="reinforce":
        return Reinforce(*args, **kwargs)
    elif algorithm_type== "reinforce_baseline":
        return ReinforceBaseline(*args, **kwargs)
    else:
        raise ValueError("Algorithm not implemented: {algorithm_type}")
