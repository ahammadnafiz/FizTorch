import numpy as np
from ..tensor import Tensor
from .module import Module
from .init_functions import xavier_uniform_, zeros_

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using Xavier initialization
        self.weight = Tensor(
            np.empty((out_features, in_features)),
            requires_grad=True
        )
        xavier_uniform_(self.weight)
        self._parameters['weight'] = self.weight

        if bias:
            self.bias = Tensor(
                np.empty(out_features),
                requires_grad=True
            )
            zeros_(self.bias)  # Initialize bias with zeros
            self._parameters['bias'] = self.bias
        else:
            self.bias = None

    def forward(self, input: Tensor) -> Tensor:
        output = input @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

class ReLU(Module):
    def forward(self, input: Tensor) -> Tensor:
        return Tensor(np.maximum(0, input.data), requires_grad=input.requires_grad)