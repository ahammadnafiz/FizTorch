import numpy as np
from typing import List, Tuple, Union, Optional

def _unbroadcast(grad, shape):
    """
    Unbroadcast gradients to match the original tensor shape.
    """
    # Handle scalars
    if not shape:
        return np.sum(grad)
        
    # Sum out the broadcasted dimensions
    axes = tuple(range(len(grad.shape) - len(shape)))  # Leading dimensions
    for i, (grad_size, shape_size) in enumerate(zip(grad.shape[len(axes):], shape)):
        if grad_size != shape_size:
            axes += (i + len(axes),)
    if axes:
        return np.sum(grad, axis=axes).reshape(shape)
    return grad

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
            gradient = Tensor(np.ones_like(self.data), requires_grad=self.requires_grad)

        if self.grad is None:
            self.grad = Tensor(gradient.data, requires_grad=self.requires_grad)
        else:
            self.grad = Tensor(self.grad.data + gradient.data, requires_grad=self.requires_grad)

        if self._grad_fn is not None:
            self._grad_fn(gradient)

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else np.array(other)
        result = Tensor(self.data + other_data, requires_grad=self.requires_grad or 
                       (isinstance(other, Tensor) and other.requires_grad))

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            def _backward(gradient):
                if self.requires_grad:
                    unbroadcast_grad = _unbroadcast(gradient.data, self.data.shape)
                    self.backward(Tensor(unbroadcast_grad, requires_grad=self.requires_grad))
                if isinstance(other, Tensor) and other.requires_grad:
                    unbroadcast_grad = _unbroadcast(gradient.data, other.data.shape)
                    other.backward(Tensor(unbroadcast_grad, requires_grad=other.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        other_data = other.data if isinstance(other, Tensor) else np.array(other)
        result = Tensor(self.data * other_data, requires_grad=self.requires_grad or 
                       (isinstance(other, Tensor) and other.requires_grad))

        if self.requires_grad or (isinstance(other, Tensor) and other.requires_grad):
            def _backward(gradient):
                if self.requires_grad:
                    grad = gradient.data * other_data
                    unbroadcast_grad = _unbroadcast(grad, self.data.shape)
                    self.backward(Tensor(unbroadcast_grad, requires_grad=self.requires_grad))
                if isinstance(other, Tensor) and other.requires_grad:
                    grad = gradient.data * self.data
                    unbroadcast_grad = _unbroadcast(grad, other.data.shape)
                    other.backward(Tensor(unbroadcast_grad, requires_grad=other.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                       requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                # If axis is None, gradient.data is a scalar
                if axis is None:
                    grad = np.full(self.data.shape, gradient.data)
                else:
                    # Expand gradient to match original shape
                    grad = np.expand_dims(gradient.data, axis=axis)
                    grad = np.broadcast_to(grad, self.data.shape)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication is only defined between tensors")
            
        result = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)

        if self.requires_grad or other.requires_grad:
            def _backward(gradient):
                if self.requires_grad:
                    grad = gradient.data @ other.data.T
                    self.backward(Tensor(grad, requires_grad=self.requires_grad))
                if other.requires_grad:
                    grad = self.data.T @ gradient.data
                    other.backward(Tensor(grad, requires_grad=other.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __neg__(self) -> 'Tensor':
        return self * -1

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self + (-other if isinstance(other, Tensor) else -np.array(other))

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return (-self) + (other if isinstance(other, Tensor) else np.array(other))

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        if isinstance(other, (int, float, np.integer)):
            return self * (1.0 / float(other))
        return self * (other ** -1)

    def __pow__(self, power: float) -> 'Tensor':
        result = Tensor(self.data ** power, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * (power * self.data ** (power - 1))
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        return self.sum(axis=axis, keepdims=keepdims) / np.prod(np.array(self.data.shape)[axis] if axis is not None else self.data.shape)

    def reshape(self, *shape) -> 'Tensor':
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                self.backward(Tensor(gradient.data.reshape(self.data.shape), 
                                   requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def exp(self) -> 'Tensor':
        result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * np.exp(self.data)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __repr__(self) -> str:
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        return self.__repr__()