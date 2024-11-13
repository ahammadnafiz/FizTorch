from ..core.tensor import Tensor
from .module import Module
import numpy as np

class Linear(Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights using Kaiming initialization
        bound = 1 / np.sqrt(in_features)
        self.weight = Tensor(
            np.random.uniform(-bound, bound, (in_features, out_features)),
            requires_grad=True
        )
        
        if bias:
            self.bias = Tensor(
                np.random.uniform(-bound, bound, (out_features,)),
                requires_grad=True
            )
        else:
            self.bias = None

    def forward(self, x):
        output = x.matmul(self.weight)
        if self.bias is not None:
            output = output + self.bias
        return output

    def extra_repr(self):
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'