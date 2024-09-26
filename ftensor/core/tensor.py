# ftensor/core/tensor.py
import numpy as np
from typing import Union, Tuple, Callable, List

class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = False, _children: Tuple['Tensor', ...] = ()):
        self.data = np.array(data, dtype=np.float32)
        self.grad: Union[np.ndarray, None] = None
        self.requires_grad = requires_grad
        self._backward: Callable[[], None] = lambda: None
        self._children = set(_children)

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.data.shape

    @property
    def dtype(self):
        return self.data.dtype

    def backward(self, gradient: Union[np.ndarray, None] = None) -> None:
        if gradient is None:
            gradient = np.ones_like(self.data)

        topo = []
        visited = set()

        def build_topo(node: 'Tensor') -> None:
            if node not in visited:
                visited.add(node)
                for child in node._children:
                    build_topo(child)
                topo.append(node)

        build_topo(self)

        self.grad = gradient
        for node in reversed(topo):
            node._backward()

    def __repr__(self) -> str:
        return f"Tensor(shape={self.shape}, dtype={self.dtype})\n{self.data}"

class FTensor:
    def __init__(self, data: Union[np.ndarray, List, Tuple], requires_grad: bool = False):
        self._tensor = Tensor(data, requires_grad=requires_grad)

    @property
    def data(self) -> np.ndarray:
        return self._tensor.data

    @property
    def grad(self) -> Union[np.ndarray, None]:
        return self._tensor.grad

    @property
    def shape(self) -> Tuple[int, ...]:
        return self._tensor.shape

    @property
    def dtype(self):
        return self._tensor.dtype

    def backward(self) -> None:
        self._tensor.backward()

    def __add__(self, other: 'FTensor') -> 'FTensor':
        return FTensor(Add.forward(None, self._tensor, other._tensor).data)

    def __mul__(self, other: 'FTensor') -> 'FTensor':
        return FTensor(Mul.forward(None, self._tensor, other._tensor).data)

    def __sub__(self, other: 'FTensor') -> 'FTensor':
        return FTensor(Sub.forward(None, self._tensor, other._tensor).data)

    def __truediv__(self, other: 'FTensor') -> 'FTensor':
        return FTensor(Div.forward(None, self._tensor, other._tensor).data)

    def sum(self, axis: Union[int, Tuple[int, ...], None] = None) -> 'FTensor':
        return FTensor(Sum.forward(None, self._tensor, axis).data)

    def mean(self, axis: Union[int, Tuple[int, ...], None] = None) -> 'FTensor':
        return FTensor(Mean.forward(None, self._tensor, axis).data)

    def dot(self, other: 'FTensor') -> 'FTensor':
        return FTensor(Dot.forward(None, self._tensor, other._tensor).data)

    def transpose(self) -> 'FTensor':
        return FTensor(Transpose.forward(None, self._tensor).data)

    def reshape(self, *new_shape):
        return FTensor(Reshape.apply(self._tensor, *new_shape))

    def log(self) -> 'FTensor':
        return FTensor(Log.forward(None, self._tensor).data)

    def exp(self) -> 'FTensor':
        return FTensor(Exp.forward(None, self._tensor).data)

    def relu(self) -> 'FTensor':
        return FTensor(ReLU.forward(None, self._tensor).data)

    def sigmoid(self) -> 'FTensor':
        return FTensor(Sigmoid.forward(None, self._tensor).data)

    def tanh(self) -> 'FTensor':
        return FTensor(Tanh.forward(None, self._tensor).data)

    def softmax(self, axis: int = -1) -> 'FTensor':
        return FTensor(Softmax.forward(None, self._tensor, axis).data)

    @property
    def T(self):
        return FTensor(self.data.T)

    def __repr__(self) -> str:
        return f"FTensor(shape={self.shape}, dtype={self.dtype})\n{self.data}"

# Move the import of operations to the end of the file
from ..ops.functional import Add, Mul, Sub, Div, Sum, Mean, Dot, Transpose, Reshape, Log, Exp, ReLU, Sigmoid, Tanh, Softmax