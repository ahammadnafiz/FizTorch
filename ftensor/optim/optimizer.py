class Optimizer:
    def __init__(self, params):
        self.params = params

    def zero_grad(self):
        for param in self.params:
            param.grad = param.grad * 0