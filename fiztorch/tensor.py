import numpy as np
from typing import List, Tuple, Union, Optional

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, float], requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)

        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.is_leaf = True

    @property
    def shape(self) -> Tuple:
        return self.data.shape

    @property
    def T(self) -> 'Tensor':
        return Tensor(self.data.T, requires_grad=self.requires_grad)

    def backward(self, gradient: Optional['Tensor'] = None) -> None:
        if not self.requires_grad:
            return

        if gradient is None:
            gradient = Tensor(np.ones_like(self.data))

        if self.grad is None:
            self.grad = gradient
        else:
            self.grad += gradient

        if self._grad_fn is not None:
            self._grad_fn(gradient)

    def __add__(self, other: 'Tensor') -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data + other_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                self.backward(gradient)
                if isinstance(other, Tensor) and other.requires_grad:
                    other.backward(gradient)
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __mul__(self, other: 'Tensor') -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else other
        result = Tensor(self.data * other_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                self.backward(gradient * other_data)
                if isinstance(other, Tensor) and other.requires_grad:
                    other.backward(gradient * self.data)
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def sum(self) -> 'Tensor':
        result = Tensor(np.sum(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                self.backward(Tensor(np.ones_like(self.data) * gradient.data))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor(self.data @ other.data, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                self.backward(gradient @ other.data.T)
                if isinstance(other, Tensor) and other.requires_grad:
                    other.backward(self.data.T @ gradient)
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"