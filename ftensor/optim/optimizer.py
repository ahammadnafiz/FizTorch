from ..core import FTensor

class Optimizer:
    def __init__(self, parameters):
        self.parameters = parameters

    def zero_grad(self):
        for param in self.parameters:
            param.grad = None

    def step(self):
        raise NotImplementedError