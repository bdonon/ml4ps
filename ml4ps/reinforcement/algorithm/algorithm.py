from abc import ABC, abstractmethod

class Algorithm(ABC):

    @abstractmethod
    def learn(*args, **kwargs):
        pass

    @abstractmethod
    def test(*args, **kwargs):
        pass

    def save_params(self, folder):
        pass

    def load_params(self, folder):
        pass
