from typing import Optional
import numpy as np
from ..tensor import Tensor
from .module import Module
from .init_functions import xavier_uniform_, ones_, zeros_

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
    
class Dropout(Module):
    """
    Dropout layer that randomly zeroes some of the elements of the input tensor
    with probability p during training.

    Args:
        p (float): Probability of setting a value to zero. Default: 0.5.
    """
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            mask = np.random.binomial(1, 1 - self.p, input.shape)
            return input * mask / (1 - self.p)
        return input
    
class BatchNorm(Module):
    """
    Batch Normalization layer to normalize the input and stabilize training.

    Args:
        num_features (int): Number of features in the input.
        eps (float): A small value to avoid division by zero. Default: 1e-5.
        momentum (float): Momentum for the moving average. Default: 0.1.
    """
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones(num_features), requires_grad=True)
        self.beta = Tensor(np.zeros(num_features), requires_grad=True)

        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)

    def forward(self, input: Tensor) -> Tensor:
        if self.training:
            batch_mean = np.mean(input.data, axis=0)
            batch_var = np.var(input.data, axis=0)

            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * batch_var

            normalized = (input.data - batch_mean) / np.sqrt(batch_var + self.eps)
        else:
            normalized = (input.data - self.running_mean) / np.sqrt(self.running_var + self.eps)

        output = self.gamma.data * normalized + self.beta.data
        result = Tensor(output, requires_grad=input.requires_grad)

        if input.requires_grad:
            def _backward(gradient: Tensor):
                grad_gamma = np.sum(gradient.data * normalized, axis=0)
                grad_beta = np.sum(gradient.data, axis=0)

                self.gamma.backward(Tensor(grad_gamma))
                self.beta.backward(Tensor(grad_beta))

                if input.requires_grad:
                    grad_input = (1 / np.sqrt(batch_var + self.eps)) * (
                        gradient.data - np.mean(gradient.data, axis=0) - normalized * np.mean(gradient.data * normalized, axis=0))
                    input.backward(Tensor(grad_input))

            result._grad_fn = _backward
            result.is_leaf = False

        return result