from typing import Optional, Tuple, Union
import numpy as np
from ..core.tensor import Tensor

class TensorOps:
    """
    Implementation of tensor operations similar to PyTorch's functionality.
    Provides utilities for broadcasting, extending dimensions, and handling gradients.
    """
    
    @staticmethod
    def max_broad(max_idx: np.ndarray, 
                 grad_out: Tensor, 
                 axis: Optional[int], 
                 out_shape: Tuple[int, ...]) -> Tensor:
        """
        Broadcasts gradients for max operation.
        
        Args:
            max_idx: Indices of maximum values
            grad_out: Output gradients
            axis: Axis along which max was computed
            out_shape: Shape of the output tensor
            
        Returns:
            Tensor containing broadcasted gradients
        """
        grad = np.zeros(out_shape)
        
        if axis is None:
            # For flattened max operation
            grad_flat = grad.reshape(-1)
            grad_flat[max_idx] = grad_out.data
        else:
            # For axis-specific max operation
            shape = max_idx.shape
            dims_strides = np.cumprod((1, *shape[::-1]))[::-1]
            
            for idx, grad_val in enumerate(grad_out.flatten()):
                # Calculate multi-dimensional indices
                indices = [
                    (idx // dims_strides[j + 1]) % shape[j]
                    for j in range(len(shape))
                ]
                indices[axis] = max_idx.flatten()[idx]
                grad[tuple(indices)] = grad_val
                
        return Tensor(grad, requires_grad=False)

    @staticmethod
    def extend(data: Tensor, 
              shape: Tuple[int, ...], 
              axis: Optional[Union[int, Tuple[int, ...]]]) -> Tensor:
        """
        Extends tensor along specified axes to match target shape.
        Similar to PyTorch's expand operation.
        
        Args:
            data: Input tensor
            shape: Target shape
            axis: Axis or axes along which to extend
            
        Returns:
            Extended tensor
        """
        if axis is None:
            return Tensor(np.tile(data.data, shape), requires_grad=data.requires_grad)
            
        # Convert single axis to tuple
        if isinstance(axis, int):
            axis = (axis,)
            
        ext_shape = tuple(
            s if idx in axis else 1 
            for idx, s in enumerate(shape)
        )
        
        return Tensor(np.tile(data.data, ext_shape), requires_grad=data.requires_grad)

    @staticmethod
    def broadcast(data: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """
        Broadcasts tensor to target shape, similar to PyTorch's broadcasting.
        
        Args:
            data: Input tensor
            shape: Target shape
            
        Returns:
            Broadcasted tensor
        """
        src_shape = data.shape
        
        # Return if shapes already match
        if src_shape == shape:
            return data
            
        # Handle different dimensionality
        if len(src_shape) != len(shape):
            broadcast_dims = tuple(range(len(src_shape) - len(shape)))
            keepdims = False
        else:
            # Find dimensions that need broadcasting
            broadcast_dims = tuple(
                idx for idx, (src, target) in enumerate(zip(src_shape, shape))
                if src != target
            )
            keepdims = True
            
        return Tensor(
            data.data.sum(axis=broadcast_dims, keepdims=keepdims),
            requires_grad=data.requires_grad
        )