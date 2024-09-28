from ..core.tensor import Tensor
from .module import Module
import numpy as np

class Linear(Module):
    def __init__(self, feature_in, feature_out):
        self.fin = feature_in
        self.fout = feature_out
        stdev = 1.0 / np.sqrt(self.fin)
        self.weight = Tensor(np.random.uniform(-stdev, stdev, (self.fin * self.fout)).reshape(self.fin, self.fout), requires_grad=True)
        self.bias = Tensor(np.random.uniform(-stdev, stdev, self.fout).reshape(self.fout), requires_grad=True)

    def forward(self, x):
        return x.matmul(self.weight) + self.bias