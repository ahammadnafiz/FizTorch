import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False, _children=()):
        self.data = np.array(data)
        self.grad = None
        self.requires_grad = requires_grad
        self._backward = lambda: None
        self._children = set(_children)

    @property
    def shape(self):
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def backward(self, gradient=None):
        if gradient is None:
            gradient = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(node):
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = gradient
        for node in reversed(topo):
            node._backward()

    def __repr__(self):
        return f"Tensor(shape={self.shape}, dtype={self.dtype})\n{self.data}"

class FTensor:
    def __init__(self, data, requires_grad=False):
        self._tensor = Tensor(data, requires_grad=requires_grad)

    @property
    def data(self):
        return self._tensor.data

    @property
    def grad(self):
        return self._tensor.grad

    @property
    def shape(self):
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    def backward(self):
        self._tensor.backward()

    def __add__(self, other):
        return FTensor(Add.forward(None, self._tensor, other._tensor).data)

    def __mul__(self, other):
        return FTensor(Mul.forward(None, self._tensor, other._tensor).data)

    def __sub__(self, other):
        return FTensor(Sub.forward(None, self._tensor, other._tensor).data)

    def __truediv__(self, other):
        return FTensor(Div.forward(None, self._tensor, other._tensor).data)

    def sum(self, axis=None):
        return FTensor(Sum.forward(None, self._tensor, axis).data)

    def mean(self, axis=None):
        return FTensor(Mean.forward(None, self._tensor, axis).data)

    def dot(self, other):
        return FTensor(Dot.forward(None, self._tensor, other._tensor).data)

    def transpose(self):
        return FTensor(Transpose.forward(None, self._tensor).data)

    def reshape(self, new_shape):
        return FTensor(Reshape.forward(self, self, new_shape))

    def log(self):
        return FTensor(Log.forward(None, self._tensor).data)

    def exp(self):
        return FTensor(Exp.forward(None, self._tensor).data)

    def relu(self):
        return FTensor(ReLU.forward(None, self._tensor).data)

    def sigmoid(self):
        return FTensor(Sigmoid.forward(None, self._tensor).data)

    def tanh(self):
        return FTensor(Tanh.forward(None, self._tensor).data)

    def softmax(self, axis=-1):
        return FTensor(Softmax.forward(None, self._tensor, axis).data)

    @property
    def T(self):
        return FTensor(self.data.T)

    def __repr__(self):
        return f"FTensor(shape={self.shape}, dtype={self.dtype})\n{self.data}"

# Move the import of operations to the end of the file
from ..ops.functional import Add, Mul, Sub, Div, Sum, Mean, Dot, Transpose, Reshape, Log, Exp, ReLU, Sigmoid, Tanh, Softmax