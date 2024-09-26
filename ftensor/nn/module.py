# ftensor/nn/module.py
from typing import List
from ..core import FTensor

class Module:
    def __init__(self):
        self._parameters: List[FTensor] = []

    def parameters(self) -> List[FTensor]:
        return self._parameters

    def forward(self, x: FTensor) -> FTensor:
        raise NotImplementedError

    def __call__(self, x: FTensor) -> FTensor:
        return self.forward(x)