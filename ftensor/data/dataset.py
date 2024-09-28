import numpy as np
from abc import ABC
from abc import abstractmethod

class Dataset(ABC):
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass