from typing import List
from .module import Module
from ..core import FTensor

class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = layers
        for layer in self.layers:
            self._parameters.extend(layer.parameters())

    def forward(self, x: FTensor) -> FTensor:
        for layer in self.layers:
            x = layer(x)
        return x