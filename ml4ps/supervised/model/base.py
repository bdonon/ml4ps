from abc import ABC

class BaseModel(ABC):
    def __init__(self) -> None:
        pass

    def init(self, rng, x):
        pass

    def loss(self, params, x, y):
        pass

    def predict(self, params, x):
        pass
