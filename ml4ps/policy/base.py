

class BasePolicy:
    def __init__(self) -> None:
        pass

    def log_prob(self, action):
        # return log probability of actions
        pass

    def sample(self, observation, seed, deterministic: bool = False):
        # return both sample action and corresponding log probabilities
        pass