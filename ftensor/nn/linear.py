# ftensor/nn/linear.py
from .module import Module
from ..core import FTensor
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weight = FTensor(np.random.randn(out_features, in_features))
        self.bias = FTensor(np.zeros(out_features))

    def forward(self, x):
        # Ensure x is properly shaped (in_features, batch_size)
        if x.shape[0] != self.weight.shape[1]:
            x = x.T
        return self.weight.dot(x) + self.bias.reshape(-1)