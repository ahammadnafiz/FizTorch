from collections import UserDict
from ..core.tensor import Tensor

class Module:
    def __init__(self):
        self.outs = []

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(ctx, *args, **kwargs): raise NotImplementedError  # noqa: E704
    def backward(ctx, *args, **kwargs): raise NotImplementedError  # noqa: E704

    def get_parameters(self):
        params = []

        def parameters(obj):
            nonlocal params
            if isinstance(obj, Tensor) and obj.requires_grad:
                params += [obj]
            if hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    parameters(v)
            if isinstance(obj, ModuleDict):
                for k, v in obj.data.items():
                    parameters(v)
            if isinstance(obj, list):
                for v in obj:
                    parameters(v)
        parameters(self)
        return params


class ModuleDict(UserDict):
    def __getattr__(self, atr):
        return self.data.get(atr, None)