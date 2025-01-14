from typing import Union, Optional
import numpy as np
from fiztorch.utils.broadcast import GradientUtils as _GradientUtils

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, float], requires_grad: bool = False):
        """
        Initialize a Tensor object.

        Parameters:
        data (Union[np.ndarray, list, float]): The data to be stored in the tensor.
        requires_grad (bool): If True, gradients will be computed for this tensor.
        """
        if isinstance(data, np.ndarray):
            self.data = data.astype(np.float64)  # Ensure float64 for numerical stability
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        self._grad_fn = None
        self.is_leaf = True

    @property
    def shape(self):
        """Return the shape of the underlying numpy array"""
        return self.data.shape
    
    def to_float32(self) -> 'Tensor':
        """Convert tensor to float32 dtype"""
        return Tensor(self.data.astype(np.float32), requires_grad=self.requires_grad)

    def to_float64(self) -> 'Tensor':
        """Convert tensor to float64 dtype"""
        return Tensor(self.data.astype(np.float64), requires_grad=self.requires_grad)

    @property
    def T(self):
        """Return a new tensor that is the transpose of this tensor"""
        result = Tensor(self.data.T, requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(gradient):
                self.backward(Tensor(gradient.data.T))
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result

    def zero_grad(self):
        """Reset the gradient to zero"""
        if self.requires_grad:
            self.grad = None
    
    def backward(self, gradient: Optional[Union['Tensor', np.ndarray]] = None) -> None:
        """
        Compute the gradient of the tensor.

        Parameters:
        gradient (Optional[Union['Tensor', np.ndarray]]): The gradient to be propagated.
        """
        if not self.requires_grad:
            return

        # Handle the case when gradient is None (implicit gradient of 1.0)
        if gradient is None:
            gradient = np.ones_like(self.data)
        elif isinstance(gradient, Tensor):
            gradient = gradient.data

        # Initialize or accumulate the gradient
        if self.grad is None:
            self.grad = Tensor(gradient)
        else:
            self.grad = Tensor(self.grad.data + gradient)  # Create new Tensor instead of modifying data

        # Propagate gradient to inputs if there's a gradient function
        if self._grad_fn is not None:
            self._grad_fn(Tensor(gradient))

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Add two tensors element-wise.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to add.

        Returns:
        Tensor: The result of the addition.
        """
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        result = Tensor(self.data + other_data, requires_grad=self.requires_grad or 
                    (isinstance(other, Tensor) and other.requires_grad))

        if result.requires_grad:
            def _backward(gradient):
                if self.requires_grad:
                    unbroadcast_grad = _GradientUtils.unbroadcast(gradient.data, self.data.shape)
                    self.backward(unbroadcast_grad)
                if isinstance(other, Tensor) and other.requires_grad:
                    unbroadcast_grad = _GradientUtils.unbroadcast(gradient.data, other.data.shape)
                    other.backward(unbroadcast_grad)
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Multiply two tensors element-wise.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to multiply.

        Returns:
        Tensor: The result of the multiplication.
        """
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        result = Tensor(self.data * other_data, requires_grad=self.requires_grad or 
                    (isinstance(other, Tensor) and other.requires_grad))

        if result.requires_grad:
            def _backward(gradient):
                if self.requires_grad:
                    grad = gradient.data * other_data
                    unbroadcast_grad = _GradientUtils.unbroadcast(grad, self.data.shape)
                    self.backward(Tensor(unbroadcast_grad))
                if isinstance(other, Tensor) and other.requires_grad:
                    grad = gradient.data * self.data
                    unbroadcast_grad = _GradientUtils.unbroadcast(grad, other.data.shape)
                    other.backward(Tensor(unbroadcast_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        """
        Perform matrix multiplication between two tensors.

        Parameters:
        other (Tensor): The tensor to multiply with.

        Returns:
        Tensor: The result of the matrix multiplication.
        """
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication is only defined between tensors")
            
        result = Tensor(self.data @ other.data, 
                       requires_grad=(self.requires_grad or other.requires_grad))

        if result.requires_grad:
            def _backward(gradient):
                if self.requires_grad:
                    self.backward(gradient @ other.T)
                if other.requires_grad:
                    other.backward(self.T @ gradient)
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __neg__(self) -> 'Tensor':
        """Return the negation of the tensor"""
        return self * -1

    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Subtract two tensors element-wise.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to subtract.

        Returns:
        Tensor: The result of the subtraction.
        """
        return self + (-other if isinstance(other, Tensor) else -np.array(other))

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Subtract this tensor from another tensor or scalar.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to subtract from.

        Returns:
        Tensor: The result of the subtraction.
        """
        return (-self) + (other if isinstance(other, Tensor) else np.array(other))

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Divide this tensor by another tensor or scalar element-wise.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to divide by.

        Returns:
        Tensor: The result of the division.
        """
        if isinstance(other, (int, float, np.integer)):
            return self * (1.0 / float(other))
        return self * (other ** -1)

    def __pow__(self, power: float) -> 'Tensor':
        """
        Raise the tensor to a power element-wise.

        Parameters:
        power (float): The power to raise the tensor to.

        Returns:
        Tensor: The result of the exponentiation.
        """
        result = Tensor(self.data ** power, requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * (power * self.data ** (power - 1))
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """
        Compute the sum of the tensor elements over a given axis.

        Parameters:
        axis (Optional[int or tuple of ints]): Axis or axes along which a sum is performed.
        keepdims (bool): If True, the axes which are reduced are left in the result as dimensions with size one.

        Returns:
        Tensor: The result of the summation.
        """
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                       requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                if axis is None:
                    grad = np.full(self.data.shape, gradient.data)
                else:
                    grad = np.expand_dims(gradient.data, axis=axis)
                    grad = np.broadcast_to(grad, self.data.shape)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        """
        Compute the mean of the tensor elements over a given axis.

        Parameters:
        axis (Optional[int or tuple of ints]): Axis or axes along which a mean is performed.
        keepdims (bool): If True, the axes which are reduced are left in the result as dimensions with size one.

        Returns:
        Tensor: The result of the mean computation.
        """
        return self.sum(axis=axis, keepdims=keepdims) / np.prod(np.array(self.data.shape)[axis] if axis is not None else self.data.shape)

    def reshape(self, *shape) -> 'Tensor':
        """
        Reshape the tensor to a new shape.

        Parameters:
        shape (tuple): The new shape.

        Returns:
        Tensor: The reshaped tensor.
        """
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                self.backward(Tensor(gradient.data.reshape(self.data.shape), 
                                   requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def exp(self) -> 'Tensor':
        """
        Compute the exponential of each element in the tensor.

        Returns:
        Tensor: The result of the exponential computation.
        """
        result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * np.exp(self.data)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def log(self) -> 'Tensor':
        """
        Compute the natural logarithm of each element in the tensor.
        
        Returns:
        Tensor: The result of the logarithm computation.
        """
        result = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data / self.data
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result

    def sin(self) -> 'Tensor':
        """
        Compute the sine of each element in the tensor.

        Returns:
        Tensor: The result of the sine computation.
        """
        result = Tensor(np.sin(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * np.cos(self.data)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def clip_grad_(self, min_val: float = None, max_val: float = None) -> None:
        """
        Clip gradients to a specified range in-place.
        
        Parameters:
        min_val (float, optional): Minimum value for the gradient
        max_val (float, optional): Maximum value for the gradient
        """
        if self.grad is not None:
            if min_val is not None:
                self.grad.data = np.maximum(self.grad.data, min_val)
            if max_val is not None:
                self.grad.data = np.minimum(self.grad.data, max_val)

    def __repr__(self) -> str:
        """Return a string representation of the tensor"""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        """Return a string representation of the tensor"""
        return self.__repr__()
