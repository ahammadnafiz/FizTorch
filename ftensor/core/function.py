# ftensor/core/function.py
from typing import List, Union, Tuple 
from .tensor import Tensor
import numpy as np

class Function:
    @staticmethod
    def apply(*args: Tensor) -> Tensor:
        ctx = Function()
        ctx.saved_tensors = [] 
        result = Function.forward(ctx, *args)
        result._children = set(args)
        result._backward = lambda: Function.backward(ctx, result.grad)
        return result

    @staticmethod
    def forward(ctx, *args: Tensor) -> Tensor:
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output: np.ndarray) -> Union[np.ndarray, Tuple[np.ndarray, ...]]:
        raise NotImplementedError

    @staticmethod
    def save_for_backward(ctx, *tensors: Tensor) -> None:
        ctx.saved_tensors = tensors

class Reshape(Function):
    @staticmethod
    def forward(ctx, tensor, *new_shape):
        ctx.save_for_backward(tensor.shape)
        return tensor.reshape(*new_shape)

    @staticmethod
    def backward(ctx, grad_output):
        original_shape, = ctx.saved_tensors
        return grad_output.reshape(original_shape), None