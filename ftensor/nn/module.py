from ..core import FTensor

class Module:
    def __init__(self):
        self._parameters = []

    def parameters(self):
        return self._parameters

    def forward(self, x):
        raise NotImplementedError

    def __call__(self, x):
        return self.forward(x)