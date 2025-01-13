from typing import Literal
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

class Loss(ABC):
    """Abstract base class for loss functions."""
    
    @abstractmethod
    def __call__(self, input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
        """Compute the loss."""
        pass

class MSELoss(Loss):
    """Mean Squared Error loss function."""
    
    def __call__(self, input: Tensor, target: Tensor, 
                 reduction: Literal['mean', 'sum', 'none'] = 'mean') -> Tensor:
        """
        Compute MSE loss between input and target.
        
        Args:
            input: Predicted values
            target: Target values
            reduction: Type of reduction to apply
            
        Returns:
            Loss tensor
        """
        diff = input - target
        if reduction == 'mean':
            return (diff * diff).sum() / diff.data.size
        elif reduction == 'sum':
            return (diff * diff).sum()
        else:  # 'none'
            return diff * diff

class CrossEntropyLoss(Loss):
    """Cross Entropy loss function with integrated softmax."""
    
    def __call__(self, input: Tensor, target: Tensor,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean') -> Tensor:
        """
        Compute cross entropy loss with integrated softmax.
        
        Args:
            input: Raw logits from the model (batch_size, num_classes)
            target: Class indices (batch_size,)
            reduction: Type of reduction to apply
            
        Returns:
            Loss tensor
        """
        batch_size = len(input.data)
        max_vals = np.max(input.data, axis=-1, keepdims=True)
        exp_x = np.exp(input.data - max_vals)
        softmax_output = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        log_softmax = np.log(softmax_output + 1e-8)
        
        nll = -log_softmax[np.arange(batch_size), target.data.astype(int)]
        
        if reduction == 'mean':
            loss_value = np.mean(nll)
        elif reduction == 'sum':
            loss_value = np.sum(nll)
        else:  # 'none'
            loss_value = nll
            
        result = Tensor(loss_value, requires_grad=input.requires_grad)
        
        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = softmax_output.copy()
                grad[np.arange(batch_size), target.data.astype(int)] -= 1
                
                if reduction == 'mean':
                    grad = grad / batch_size
                
                if not np.isscalar(gradient.data):
                    grad = grad * gradient.data.reshape(-1, 1)
                else:
                    grad = grad * gradient.data
                    
                input.backward(Tensor(grad))
                
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result

# Convenience functions for direct use
def relu(input: Tensor) -> Tensor:
    """Convenience function for ReLU activation."""
    return ReLU()(input)

def sigmoid(input: Tensor) -> Tensor:
    """Convenience function for Sigmoid activation."""
    return Sigmoid()(input)

def tanh(input: Tensor) -> Tensor:
    """Convenience function for Tanh activation."""
    return Tanh()(input)

def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Convenience function for Softmax activation."""
    return Softmax(dim=dim)(input)

def mse_loss(input: Tensor, target: Tensor, 
             reduction: Literal['mean', 'sum', 'none'] = 'mean') -> Tensor:
    """Convenience function for MSE loss."""
    return MSELoss()(input, target, reduction)

def cross_entropy(input: Tensor, target: Tensor,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean') -> Tensor:
    """Convenience function for Cross Entropy loss."""
    return CrossEntropyLoss()(input, target, reduction)