from .module import Module
from ..core import Tensor
import numpy as np

class Softmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        e = (x - x.max(axis=self.dim, keepdims=True)).exp()
        s = e.sum(axis=self.dim)
        sm = e.div(s)
        return sm

class LogSoftmax(Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        e = (x - x.max(axis=self.dim, keepdims=True)).exp()
        s = e.sum(axis=self.dim)
        sm = e.log() - s.log()
        return sm

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        self.normalized_shape = normalized_shape
        self.weight = Tensor.ones(normalized_shape, requires_grad=True)
        self.bias = Tensor.zeros(normalized_shape, requires_grad=True)
        self.eps = eps

    def forward(self, x):
        xe = x - x.mean(axis=-1)
        return xe.div(((xe * xe).mean(axis=-1) + self.eps) ** 0.5) * self.weight + self.bias

class Embedding(Module):
    def __init__(self, n_embd, embd_dim):
        super().__init__()
        self.embd_dim = embd_dim
        self.weight = Tensor.ones((n_embd, embd_dim), requires_grad=True)

    def forward(self, idx):
        return self.weight._embed(idx, n_embd=self.embd_dim)
    
class ReLU(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * (x > 0)  # Apply ReLU: max(0, x)

class Tanh(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (x.exp() - (-x).exp()).div((x.exp() + (-x).exp()))  # Apply tanh: (e^x - e^-x) / (e^x + e^-x)

class Sigmoid(Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return (1.0 + (-x).exp()).div(1.0)  # Apply sigmoid: 1 / (1 + e^-x)
    
    
def one_hot(label, num_classes):
    shape = label.shape[0], num_classes
    y = np.zeros(shape)
    y_ptr = y.reshape((-1,))
    idx = label.flatten() + np.arange(0, (np.prod(shape)), shape[1])
    y_ptr[idx] = 1
    return y


class CrossEntropy(Module):
    def __init__(self, num_class):
        super().__init__()
        self.num_class = num_class

    def forward(self, y, target):
        T = Tensor(one_hot(target, self.num_class), requires_grad=False)
        return (y * T).sum() * -1