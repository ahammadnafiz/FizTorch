from typing import Iterator
from ..tensor import Tensor

class SGD:
    def __init__(self, parameters: Iterator[Tensor], lr: float = 0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self) -> None:
        for param in self.parameters:
            if param.grad is not None:
                param.grad = None

    def step(self):
        for param in self.parameters:
            if param.grad is not None:
                # print(f"Before update: {param.data}")
                param.data -= self.lr * param.grad.data
                # param.grad = None  # Zero out the gradient after update
                print(f"After update: {param.data}")