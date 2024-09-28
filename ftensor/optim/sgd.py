from .optimizer import Optimizer

class SGD(Optimizer):
    def __init__(self, params, lr):
        super().__init__(params)
        self.lr = lr

    def step(self):
        for param in self.params:
            param -= param.grad * self.lr
