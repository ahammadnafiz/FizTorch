from .tensor import Tensor
import numpy as np

class Function:
    @staticmethod
    def apply(*args) -> Tensor:
        ctx = Function()
        ctx.saved_tensors = [] 
        result = Function.forward(ctx, *args)
        result._children = set(args)
        result._backward = lambda: Function.backward(ctx, result.grad)
        return result

    @staticmethod
    def forward(ctx, *args):
        raise NotImplementedError

    @staticmethod
    def backward(ctx, grad_output):
        raise NotImplementedError

    @staticmethod
    def save_for_backward(ctx, *tensors):
        ctx.saved_tensors = tensors