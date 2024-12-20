from typing import Iterator
from ..tensor import Tensor

class SGD:
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None  # Simply set to None instead of checking

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                # Update parameters using gradient
                param.data -= self.lr * param.grad.data