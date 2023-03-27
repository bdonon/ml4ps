from abc import ABC, abstractmethod

class Algorithm(ABC):

    @abstractmethod
    def learn(*args, **kwargs):
        pass

    @abstractmethod
    def test(*args, **kwargs):
        pass
