from .tensor import Tensor
import numpy as np

class Function:
    def __init__(self):
        self.outs = []

    @classmethod
    def call(cls, *args, **kwargs):
        ctx = cls()
        ctx.parents = args
        ctx.requires_grad = any(ar.requires_grad for ar in args if isinstance(ar, Tensor))
        return ctx.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): 
        raise NotImplementedError
    def backward(ctx, *args, **kwargs): 
        raise NotImplementedError