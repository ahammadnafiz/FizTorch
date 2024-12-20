from typing import Dict, Iterator, Tuple
from ..tensor import Tensor

class Module:
    """
    Base class for all neural network modules.

    Your models should also subclass this class. Modules can also contain other
    Modules, allowing to nest them in a tree structure. You can assign the submodules
    as regular attributes.

    Attributes:
        _parameters (Dict[str, Tensor]): A dictionary mapping parameter names to Tensors.
        training (bool): Boolean flag indicating whether the module is in training mode.

    Methods:
        __init__():
            Initializes internal Module state, including parameters and training mode.
        
        parameters() -> Iterator[Tensor]:
            Returns an iterator over module parameters.
        
        zero_grad() -> None:
            Sets gradients of all parameters to zero.
        
        __call__(*args, **kwargs):
            Calls the forward method. This makes the module callable.
        
        forward(*args, **kwargs):
            Defines the computation performed at every call. Should be overridden by all subclasses.
        
        train(mode: bool = True) -> 'Module':
            Sets the module in training mode.
        
        eval() -> 'Module':
            Sets the module in evaluation mode.
    """
    def __init__(self):
        self._parameters: Dict[str, Tensor] = {}
        self.training: bool = True

    def parameters(self) -> Iterator[Tensor]:
        for param in self._parameters.values():
            yield param

    def zero_grad(self) -> None:
        for param in self.parameters():
            if param.grad is not None:
                param.grad.data.fill_(0)

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def train(self, mode: bool = True) -> 'Module':
        self.training = mode
        return self

    def eval(self) -> 'Module':
        return self.train(False)