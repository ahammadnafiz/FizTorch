import numpy as np
from ..tensor import Tensor
from .module import Module
from .init_functions import xavier_uniform_, zeros_

class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Initialize weights using Xavier initialization
        self.weight = Tensor(
            np.empty((out_features, in_features)),
            requires_grad=True
        )
        xavier_uniform_(self.weight)
        self._parameters['weight'] = self.weight

        if bias:
            self.bias = Tensor(
                np.empty(out_features),
                requires_grad=True
            )
            zeros_(self.bias)  # Initialize bias with zeros
            self._parameters['bias'] = self.bias
        else:
            self.bias = None

    def forward(self, x):
        # Ensure tensors have requires_grad set
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        if not self.weight.requires_grad:
            self.weight = Tensor(self.weight.data, requires_grad=True)
            
        if not self.bias.requires_grad:
            self.bias = Tensor(self.bias.data, requires_grad=True)
        
        # Use matmul and ensure gradient tracking
        out = x @ self.weight.T + self.bias
        
        # Create backward function
        def _backward(gradient):
            if self.weight.requires_grad:
                weight_grad = gradient.data.T @ x.data
                self.weight.backward(Tensor(weight_grad, requires_grad=True))
            if self.bias.requires_grad:
                bias_grad = gradient.data.sum(axis=0)
                self.bias.backward(Tensor(bias_grad, requires_grad=True))
                
        out._grad_fn = _backward
        out.is_leaf = False
        return out

class ReLU(Module):
    def forward(self, x):
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        
        if x.requires_grad:
            def _backward(gradient):
                grad = gradient.data * (x.data > 0)
                x.backward(Tensor(grad, requires_grad=x.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        return result