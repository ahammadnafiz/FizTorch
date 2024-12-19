from typing import Dict, Iterator, Tuple
from ..tensor import Tensor

class Module:
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self.training: bool = True

    def parameters(self) -> Iterator[Tensor]:
        for param in self._parameters.values():
            yield param

    def zero_grad(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        return self

    def eval(self) -> 'Module':
        return self.train(False)