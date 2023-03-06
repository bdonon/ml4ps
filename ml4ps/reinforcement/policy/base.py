from abc import ABC
from typing import Tuple, Dict

from ml4ps.h2mg import H2MG


class BasePolicy(ABC):
    def __init__(self) -> None:
        pass

    def init(self, rng, observation):
        pass

    def log_prob(self, params, observation, action):
        # return log probability of actions
        pass

    def sample(self, params, observation, seed=0, deterministic: bool = False) -> Tuple[H2MG, float, Dict]:
        # return both sample action and corresponding log probabilities
        pass