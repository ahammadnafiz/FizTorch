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
    
class BCELoss(Loss):
    """Binary Cross Entropy loss function with built-in sigmoid activation."""
    
    def __call__(self, input: Tensor, target: Tensor,
                 reduction: Literal['mean', 'sum', 'none'] = 'mean') -> Tensor:
        """
        Compute binary cross entropy loss with integrated sigmoid.
        
        Args:
            input: Raw logits from the model (batch_size,) or (batch_size, 1)
            target: Target values in range [0,1] with same shape as input
            reduction: Type of reduction to apply
            
        Returns:
            Loss tensor
        """
        # Ensure input and target have same shape
        input_data = input.data.reshape(-1)
        target_data = target.data.reshape(-1)
        
        if input_data.shape != target_data.shape:
            raise ValueError(f"Target shape {target.data.shape} must match input shape {input.data.shape}")
            
        # Apply sigmoid with numerical stability
        x = input_data
        x_safe = x * (x >= 0) - x * (x < 0)
        exp_x = np.exp(x_safe)
        
        # Compute sigmoid
        sigmoid_x = np.where(x >= 0, 
                           exp_x / (1 + exp_x),
                           1 / (1 + np.exp(-x_safe)))
        
        # Clip values for numerical stability
        eps = 1e-12
        sigmoid_x = np.clip(sigmoid_x, eps, 1 - eps)
        
        # Compute binary cross entropy
        bce = -target_data * np.log(sigmoid_x) - (1 - target_data) * np.log(1 - sigmoid_x)
        
        # Apply reduction
        if reduction == 'mean':
            loss_value = np.mean(bce)
        elif reduction == 'sum':
            loss_value = np.sum(bce)
        else:  # 'none'
            loss_value = bce
            
        result = Tensor(loss_value, requires_grad=input.requires_grad)
        
        if input.requires_grad:
            def _backward(gradient: Tensor) -> None:
                grad = (sigmoid_x - target_data)
                
                if reduction == 'mean':
                    grad = grad / len(grad)
                    
                if not np.isscalar(gradient.data):
                    grad = grad * gradient.data
                else:
                    grad = grad * gradient.data
                    
                # Reshape grad back to original input shape
                grad = grad.reshape(input.data.shape)
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

def binary_cross_entropy(input: Tensor, target: Tensor,
                        reduction: Literal['mean', 'sum', 'none'] = 'mean') -> Tensor:
    """Convenience function for Binary Cross Entropy loss."""
    return BCELoss()(input, target, reduction)