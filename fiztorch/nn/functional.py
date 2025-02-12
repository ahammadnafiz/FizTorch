from typing import Literal
from ..tensor import Tensor
from .activations import (
    ReLU, 
    LeakyReLU, 
    ELU,
    SELU,
    Sigmoid, 
    Tanh, 
    Softmax)
from .losses import (
    MSELoss, 
    CrossEntropyLoss, 
    BCELoss)

# Convenience functions for direct use
def relu(input: Tensor) -> Tensor:
    """Convenience function for ReLU activation."""
    return ReLU()(input)

def leaky_relu(input: Tensor, negative_slope: float = 0.01) -> Tensor:
    """Convenience function for LeakyReLU activation."""
    return LeakyReLU(negative_slope)(input)

def elu(input: Tensor, alpha: float = 1.0) -> Tensor:
    """Convenience function for ELU activation."""
    return ELU(alpha)(input)

def selu(input: Tensor) -> Tensor:
    """Convenience function for SELU activation."""
    return SELU()(input)

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