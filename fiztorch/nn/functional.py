import numpy as np
from ..tensor import Tensor
from typing import Optional

def relu(input: Tensor) -> Tensor:
    """Applies the rectified linear unit function."""
    return Tensor(np.maximum(0, input.data), requires_grad=input.requires_grad)

def sigmoid(input: Tensor) -> Tensor:
    """Applies the sigmoid function."""
    return Tensor(1 / (1 + np.exp(-input.data)), requires_grad=input.requires_grad)

def tanh(input: Tensor) -> Tensor:
    """Applies the hyperbolic tangent function."""
    return Tensor(np.tanh(input.data), requires_grad=input.requires_grad)

# def softmax(input: Tensor, dim: int = -1) -> Tensor:
#     """Applies the softmax function."""
#     exp_x = np.exp(input.data - np.max(input.data, axis=dim, keepdims=True))
#     return Tensor(exp_x / np.sum(exp_x, axis=dim, keepdims=True), requires_grad=input.requires_grad)

def mse_loss(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """Mean squared error loss."""
    diff = input - target
    if reduction == 'mean':
        return (diff * diff).sum() / diff.data.size
    elif reduction == 'sum':
        return (diff * diff).sum()
    else:
        return diff * diff

def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
    """
    Cross entropy loss that handles both integer class indices and one-hot encoded labels.
    
    Args:
        input (Tensor): Logits from the model, shape (batch_size, num_classes).
        target (Tensor): Targets, either:
            - Integer class indices, shape (batch_size,)
            - One-hot encoded labels, shape (batch_size, num_classes)
        reduction (str): Specifies the reduction to apply. Options: 'mean', 'sum', or 'none'.

    Returns:
        Tensor: Computed cross-entropy loss.
    """
    # Compute softmax probabilities
    log_probs = softmax(input, dim=-1)
    
    # Check if target is one-hot encoded (2D) or class indices (1D)
    if target.data.ndim == 2:
        # Convert one-hot encoded targets to integer class indices
        target_indices = np.argmax(target.data, axis=1)
    elif target.data.ndim == 1:
        target_indices = target.data.astype(int)
    else:
        raise ValueError("Target must be 1D (class indices) or 2D (one-hot encoded).")
    
    # Negative log likelihood
    nll = -log_probs.data[np.arange(len(target_indices)), target_indices]
    
    # Apply reduction
    if reduction == 'mean':
        return Tensor(np.mean(nll), requires_grad=input.requires_grad)
    elif reduction == 'sum':
        return Tensor(np.sum(nll), requires_grad=input.requires_grad)
    elif reduction == 'none':
        return Tensor(nll, requires_grad=input.requires_grad)
    else:
        raise ValueError("Invalid reduction type. Must be 'mean', 'sum', or 'none'.")


def softmax(input: Tensor, dim: int = -1) -> Tensor:
    """Applies the softmax function with proper gradient computation."""
    max_val = np.max(input.data, axis=dim, keepdims=True)
    exp_x = np.exp(input.data - max_val)
    softmax_output = exp_x / np.sum(exp_x, axis=dim, keepdims=True)
    result = Tensor(softmax_output, requires_grad=input.requires_grad)
    
    if input.requires_grad:
        def _backward(gradient):
            # Proper softmax gradient computation
            s = softmax_output
            grad = s * (gradient.data - np.sum(gradient.data * s, axis=dim, keepdims=True))
            input.backward(Tensor(grad, requires_grad=input.requires_grad))
        result._grad_fn = _backward
        result.is_leaf = False
    
    return result

# def cross_entropy(input: Tensor, target: Tensor, reduction: str = 'mean') -> Tensor:
#     """Cross entropy loss with proper gradient computation."""
#     batch_size = len(input.data)
#     softmax_output = softmax(input, dim=-1)
    
#     # Compute cross entropy loss
#     log_probs = np.log(softmax_output.data + 1e-8)  # Add small epsilon for numerical stability
#     nll = -log_probs[np.arange(batch_size), target.data.astype(int)]
    
#     if reduction == 'mean':
#         loss_value = np.mean(nll)
#     elif reduction == 'sum':
#         loss_value = np.sum(nll)
#     else:
#         loss_value = nll
        
#     result = Tensor(loss_value, requires_grad=input.requires_grad)
    
#     if input.requires_grad:
#         def _backward(gradient):
#             # Compute gradient for cross entropy loss
#             grad = softmax_output.data.copy()
#             grad[np.arange(batch_size), target.data.astype(int)] -= 1
#             if reduction == 'mean':
#                 grad = grad / batch_size
#             elif reduction == 'sum':
#                 grad = grad
#             input.backward(Tensor(grad * gradient.data, requires_grad=input.requires_grad))
#         result._grad_fn = _backward
#         result.is_leaf = False
    
#     return result