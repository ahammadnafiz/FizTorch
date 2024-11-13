from .module import Module
from ..core import Tensor
import numpy as np

class Softmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_x = x.max(axis=self.dim, keepdims=True)
        exp_x = (x - max_x).exp()
        sum_exp_x = exp_x.sum(axis=self.dim, keepdims=True)
        return exp_x.div(sum_exp_x)

class LogSoftmax(Module):
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        max_x = x.max(axis=self.dim, keepdims=True)
        exp_x = (x - max_x).exp()
        sum_exp_x = exp_x.sum(axis=self.dim, keepdims=True)
        return x - max_x - sum_exp_x.log()

class LayerNorm(Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.weight = Tensor.ones(normalized_shape)
        self.bias = Tensor.zeros(normalized_shape)

    def forward(self, x):
        mean = x.mean(axis=-1, keepdims=True)
        var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.eps).pow(0.5)
        return x_norm * self.weight + self.bias

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Tensor(
            np.random.normal(0, 0.02, (num_embeddings, embedding_dim)), 
            requires_grad=True
        )

    def forward(self, indices):
        if not isinstance(indices, Tensor):
            indices = Tensor(indices, requires_grad=False)
        return self.weight[indices]

class ReLU(Module):
    def forward(self, x):
        return x.relu()

class Tanh(Module):
    def forward(self, x):
        return (x.exp() - (-x).exp()).div(x.exp() + (-x).exp())

class Sigmoid(Module):
    def forward(self, x):
        return Tensor(1.0).div(Tensor(1.0) + (-x).exp())

def one_hot(labels, num_classes):
    if isinstance(labels, Tensor):
        labels = labels.data
    shape = (labels.shape[0], num_classes)
    one_hot = np.zeros(shape)
    np.put_along_axis(one_hot, labels.reshape(-1, 1), 1, axis=1)
    return one_hot

class CrossEntropyLoss(Module):
    def __init__(self, num_classes=None):
        super().__init__()
        self.num_classes = num_classes
        self.log_softmax = LogSoftmax(dim=-1)

    def forward(self, input, target):
        if self.num_classes is None:
            self.num_classes = input.shape[-1]
        log_prob = self.log_softmax(input)
        target = Tensor(one_hot(target, self.num_classes), requires_grad=False)
        return -(log_prob * target).sum() / target.shape[0]  # Mean reduction