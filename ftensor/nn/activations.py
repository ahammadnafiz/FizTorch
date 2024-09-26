# ftensor/nn/activations.py
from .module import Module
from ..core import FTensor
from typing import Callable

class Activation(Module):
    def __init__(self, activation_fn: Callable[[FTensor], FTensor]):
        super().__init__()
        self.activation_fn = activation_fn

    def forward(self, x: FTensor) -> FTensor:
        return self.activation_fn(x)

class ReLU(Activation):
    def __init__(self):
        super().__init__(lambda x: x.relu())

class Sigmoid(Activation):
    def __init__(self):
        super().__init__(lambda x: x.sigmoid())

class Tanh(Activation):
    def __init__(self):
        super().__init__(lambda x: x.tanh())

class Softmax(Activation):
    def __init__(self, axis: int = -1):
        super().__init__(lambda x: x.softmax(axis=axis))