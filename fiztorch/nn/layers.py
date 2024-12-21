import numpy as np
from ..tensor import Tensor
from .module import Module
from .init_functions import xavier_uniform_, zeros_

class Linear(Module):
    """
    A fully connected linear layer.

    Args:
        in_features (int): Size of each input sample.
        out_features (int): Size of each output sample.
        bias (bool): If set to False, the layer will not learn an additive bias. Default: True.
    """
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
        """
        Forward pass for the linear layer.

        Args:
            x (Tensor): Input tensor of shape (N, in_features) where N is the batch size.

        Returns:
            Tensor: Output tensor of shape (N, out_features).
        """
        # Ensure input is a Tensor with requires_grad set
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        if not self.weight.requires_grad:
            self.weight = Tensor(self.weight.data, requires_grad=True)
            
        if self.bias is not None and not self.bias.requires_grad:
            self.bias = Tensor(self.bias.data, requires_grad=True)
        
        # Compute the linear transformation
        out = x @ self.weight.T + self.bias
        
        # Define the backward function for gradient computation
        def _backward(gradient):
            if self.weight.requires_grad:
                weight_grad = gradient.data.T @ x.data
                self.weight.backward(Tensor(weight_grad, requires_grad=True))
            if self.bias is not None and self.bias.requires_grad:
                bias_grad = gradient.data.sum(axis=0)
                self.bias.backward(Tensor(bias_grad, requires_grad=True))
                
        out._grad_fn = _backward
        out.is_leaf = False
        return out

class ReLU(Module):
    """
    Applies the rectified linear unit function element-wise: ReLU(x) = max(0, x)
    """
    def forward(self, x):
        """
        Forward pass for the ReLU activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor where each element is the result of applying ReLU to the corresponding element of the input tensor.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        # Apply ReLU activation function
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        
        if x.requires_grad:
            # Define the backward function for gradient computation
            def _backward(gradient):
                grad = gradient.data * (x.data > 0)
                x.backward(Tensor(grad, requires_grad=x.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        return result