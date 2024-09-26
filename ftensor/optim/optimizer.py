# ftensor/optim/optimizer.py
from typing import List
from ..core import FTensor

class Optimizer:
    def __init__(self, parameters: List[FTensor]):
        self.parameters = parameters

    def zero_grad(self) -> None:
        for param in self.parameters:
            param.grad = None

    def step(self) -> None:
        raise NotImplementedError