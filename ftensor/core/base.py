# ftensor/core/base.py
from abc import ABC, abstractmethod

class TensorBase(ABC):
    @property
    @abstractmethod
    def data(self):
        pass

    @property
    @abstractmethod
    def shape(self):
        pass

    @property
    @abstractmethod
    def size(self):
        pass

    @abstractmethod
    def __add__(self, other):
        pass

    @abstractmethod
    def __mul__(self, other):
        pass

    @abstractmethod
    def __sub__(self, other):
        pass

    @abstractmethod
    def __truediv__(self, other):
        pass

    @abstractmethod
    def sum(self, axis=None):
        pass

    @abstractmethod
    def dot(self, other):
        pass

    @abstractmethod
    def transpose(self):
        pass

    @abstractmethod
    def flatten(self):
        pass

    @abstractmethod
    def reshape(self, new_shape):
        pass

    @property
    @abstractmethod
    def dtype(self):
        pass