from ml4ps.supervised.problem.ps_problem import *
from ml4ps.supervised.problem.library import *


def get_problem(problem_type: str, *problem_args, **problem_kwargs):
    if problem_type == "ACPowerFlowProxyPandapower":
        return ACPowerFlowProxyPandapower(*problem_args, **problem_kwargs)
    elif problem_type == "ACPowerFlowProxyPypowsybl":
        return ACPowerFlowProxyPypowsybl(*problem_args, **problem_kwargs)
    else:
        raise NotImplementedError("No existing model of type {}.".format(problem_type))
