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

        # print(f"Weight initialized: {self.weight}")
        # if bias:
        #     print(f"Bias initialized: {self.bias}")


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

class LeakyReLU(Module):
    """
    Applies the Leaky Rectified Linear Unit function element-wise: LeakyReLU(x) = x if x > 0 else alpha * x
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initialize LeakyReLU.

        Args:
            negative_slope (float): Slope for negative inputs. Default: 0.01.
        """
        super().__init__()
        self.negative_slope = negative_slope

    def forward(self, x):
        """
        Forward pass for the LeakyReLU activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor where each element is the result of applying LeakyReLU to the corresponding element of the input tensor.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)

        # Apply LeakyReLU activation function
        result = Tensor(np.where(x.data > 0, x.data, self.negative_slope * x.data), requires_grad=x.requires_grad)

        if x.requires_grad:
            # Define the backward function for gradient computation
            def _backward(gradient):
                grad = np.where(x.data > 0, gradient.data, self.negative_slope * gradient.data)
                x.backward(Tensor(grad, requires_grad=x.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        return result

class Sigmoid(Module):
    """
    Applies the sigmoid activation function element-wise: Sigmoid(x) = 1 / (1 + exp(-x))
    """
    def forward(self, x):
        """
        Forward pass for the sigmoid activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor where each element is the result of applying sigmoid to the corresponding element of the input tensor.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)

        # Apply sigmoid activation function
        data = 1 / (1 + np.exp(-x.data))
        result = Tensor(data, requires_grad=x.requires_grad)

        if x.requires_grad:
            # Define the backward function for gradient computation
            def _backward(gradient):
                grad = gradient.data * data * (1 - data)
                x.backward(Tensor(grad, requires_grad=x.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        return result

class Softmax(Module):
    """
    Applies the softmax activation function to the input tensor along the specified axis.
    """
    def __init__(self, axis=-1):
        super().__init__()
        self.axis = axis

    def forward(self, x):
        """
        Forward pass for the softmax activation function.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor where softmax is applied along the specified axis.
        """
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)

        # Apply softmax activation function
        exp_data = np.exp(x.data - np.max(x.data, axis=self.axis, keepdims=True))
        data = exp_data / np.sum(exp_data, axis=self.axis, keepdims=True)
        result = Tensor(data, requires_grad=x.requires_grad)

        if x.requires_grad:
            # Define the backward function for gradient computation
            def _backward(gradient):
                s = data.reshape(-1, 1)
                jacobian = np.diagflat(s) - np.outer(s, s)
                grad = gradient.data @ jacobian
                x.backward(Tensor(grad, requires_grad=x.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        return result