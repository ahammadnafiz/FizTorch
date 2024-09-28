import numpy as np
from ..core import FTensor

class Dataset:
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x, y = self.data[idx]
        return FTensor(x), FTensor(y)