from collections import UserDict
from ..core.tensor import Tensor

class Module:
    def __init__(self):
        self._parameters = []
        self.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, mode=True):
        self.training = mode
        return self

    def eval(self):
        return self.train(False)

    def parameters(self):
        """Returns an iterator over module parameters"""
        if not hasattr(self, '_parameters'):
            self._parameters = []
        return iter(self.get_parameters())

    def get_parameters(self):
        params = []
        visited = set()  # To keep track of visited modules

        def _get_parameters(obj):
            if id(obj) in visited:  # Check if we've already visited this object
                return
            visited.add(id(obj))  # Mark this object as visited

            if isinstance(obj, Tensor) and obj.requires_grad:
                params.append(obj)
            elif isinstance(obj, Module):
                # Instead of calling get_parameters, we directly check for parameters
                if hasattr(obj, '_parameters'):
                    params.extend(obj._parameters)  # Directly access _parameters
                # Iterate through children or attributes
                for attr in vars(obj).values():
                    _get_parameters(attr)
            elif isinstance(obj, (list, tuple)):
                for item in obj:
                    _get_parameters(item)
            elif isinstance(obj, ModuleDict):
                for k, v in obj.data.items():
                    _get_parameters(v)
            elif hasattr(obj, "__dict__"):
                for k, v in obj.__dict__.items():
                    if not k.startswith('_'):
                        _get_parameters(v)

        _get_parameters(self)
        return params

    def zero_grad(self):
        """Zeros out the gradients of all parameters"""
        for p in self.parameters():
            if p.grad is not None:
                p.grad.data.fill_(0)

class ModuleDict(UserDict):
    def __getattr__(self, key):
        try:
            return self.data[key]
        except KeyError:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{key}'")