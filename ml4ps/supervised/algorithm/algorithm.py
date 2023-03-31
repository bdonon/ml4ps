from abc import ABC, abstractmethod

class SupervisedAlgorithm(ABC):

    @abstractmethod
    def learn(*args, **kwargs):
        pass

    @abstractmethod
    def test(*args, **kwargs):
        pass
