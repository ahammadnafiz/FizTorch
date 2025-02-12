import numpy as np
from abc import ABC, abstractmethod
from ..tensor import Tensor

class Activation(ABC):
    """Abstract base class for activation functions."""
    
    @abstractmethod
    def __call__(self, input: Tensor) -> Tensor:
        """Apply the activation function."""
        pass

class ReLU(Activation):
    """Rectified Linear Unit activation function."""
    
    def __call__(self, input: Tensor) -> Tensor:
        output_data = np.maximum(0, input.data)
        result = Tensor(output_data, requires_grad=input.requires_grad)
        
        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = gradient.data * (input.data > 0).astype(np.float64)
                input.backward(Tensor(grad))
            
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
class LeakyReLU(Activation):
    """
    Leaky ReLU activation function: LeakyReLU(x) = x if x > 0 else alpha * x
    """
    def __init__(self, negative_slope: float = 0.01):
        """
        Initialize LeakyReLU.

        Args:
            negative_slope (float): Slope for negative inputs. Default: 0.01.
        """
        self.negative_slope = negative_slope

    def __call__(self, input: Tensor) -> Tensor:
        output_data = np.where(input.data > 0, input.data, self.negative_slope * input.data)
        result = Tensor(output_data, requires_grad=input.requires_grad)

        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = np.where(input.data > 0, gradient.data, self.negative_slope * gradient.data)
                input.backward(Tensor(grad))
            
            result._grad_fn = _backward
            result.is_leaf = False

        return result

class ELU(Activation):
    """
    Exponential Linear Unit activation function: ELU(x) = x if x > 0 else alpha * (exp(x) - 1)
    """
    def __init__(self, alpha: float = 1.0):
        """
        Initialize ELU.

        Args:
            alpha (float): Alpha value for negative inputs. Default: 1.0.
        """
        self.alpha = alpha

    def __call__(self, input: Tensor) -> Tensor:
        output_data = np.where(input.data > 0, input.data, self.alpha * (np.exp(input.data) - 1))
        result = Tensor(output_data, requires_grad=input.requires_grad)

        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = np.where(input.data > 0, gradient.data, self.alpha * np.exp(input.data) * gradient.data)
                input.backward(Tensor(grad))
            
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
class SELU(Activation):
    """
    Scaled Exponential Linear Unit activation function: SELU(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
    """
    def __init__(self, alpha: float = 1.6732632423543772848170429916717, scale: float = 1.0507009873554804934193349852946):
        """
        Initialize SELU.

        Args:
            alpha (float): Alpha value for negative inputs. Default: 1.6732632423543772848170429916717.
            scale (float): Scale value for the output. Default: 1.0507009873554804934193349852946.
        """
        self.alpha = alpha
        self.scale = scale

    def __call__(self, input: Tensor) -> Tensor:
        output_data = self.scale * np.where(input.data > 0, input.data, self.alpha * (np.exp(input.data) - 1))
        result = Tensor(output_data, requires_grad=input.requires_grad)

        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = self.scale * np.where(input.data > 0, gradient.data, self.alpha * np.exp(input.data) * gradient.data)
                input.backward(Tensor(grad))
            
            result._grad_fn = _backward
            result.is_leaf = False

        return result

class Sigmoid(Activation):
    """Sigmoid activation function."""
    
    def __call__(self, input: Tensor) -> Tensor:
        sigmoid_data = 1 / (1 + np.exp(-input.data))
        result = Tensor(sigmoid_data, requires_grad=input.requires_grad)

        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = gradient.data * sigmoid_data * (1 - sigmoid_data)
                input.backward(Tensor(grad))
            
            result._grad_fn = _backward
            result.is_leaf = False

        return result

class Tanh(Activation):
    """Hyperbolic tangent activation function."""
    
    def __call__(self, input: Tensor) -> Tensor:
        tanh_data = np.tanh(input.data)
        result = Tensor(tanh_data, requires_grad=input.requires_grad)

        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = gradient.data * (1 - tanh_data ** 2)
                input.backward(Tensor(grad))
            
            result._grad_fn = _backward
            result.is_leaf = False

        return result

class Softmax(Activation):
    """Softmax activation function."""
    
    def __init__(self, dim: int = -1):
        """
        Initialize Softmax activation.
        
        Args:
            dim: Dimension along which to compute softmax
        """
        self.dim = dim
    
    def __call__(self, input: Tensor) -> Tensor:
        max_val = np.max(input.data, axis=self.dim, keepdims=True)
        exp_x = np.exp(input.data - max_val)
        softmax_output = exp_x / np.sum(exp_x, axis=self.dim, keepdims=True)
        result = Tensor(softmax_output, requires_grad=input.requires_grad)
        
        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                s = softmax_output
                grad = s * (gradient.data - np.sum(gradient.data * s, axis=self.dim, keepdims=True))
                input.backward(Tensor(grad, requires_grad=input.requires_grad))
            
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result