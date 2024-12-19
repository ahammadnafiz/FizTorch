from typing import List, Iterator
from .module import Module
from ..tensor import Tensor

class Sequential(Module):
    def __init__(self, *layers: Module):
        super().__init__()
        self.layers = list(layers)
        for idx, layer in enumerate(self.layers):
            self._parameters[f'layer_{idx}'] = layer

    def forward(self, input: Tensor) -> Tensor:
        for layer in self.layers:
            input = layer(input)
        return input

    def parameters(self) -> Iterator[Tensor]:
        for layer in self.layers:
            yield from layer.parameters()

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def append(self, module: Module) -> None:
        self.layers.append(module)
        self._parameters[f'layer_{len(self.layers)-1}'] = module