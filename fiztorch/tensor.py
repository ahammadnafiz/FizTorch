from typing import Union, Optional, List
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
            self.data = data
        else:
            self.data = np.array(data) # Convert to numpy array

        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None # Gradient of the tensor with respect to some scalar value
        self._grad_fn = None # Function to compute the gradient of the tensor
        self.is_leaf = True # Indicates whether the tensor is a leaf node in the computational graph

        # TODO: TRACK PARENT-CHILD RELATIONSHIPS FOR BACKPROPAGATION
        self.parents: List[Tensor] = []
        self._grad_accumulated = False
        self.children: List[Tensor] = []

    def backward(self, gradient: Optional[Union['Tensor', np.ndarray]] = None) -> None:
        """
        Compute the gradient of the tensor using topological sort.

        Parameters:
        gradient (Optional[Union['Tensor', np.ndarray]]): The gradient to be propagated.
        """
        if not self.requires_grad:
            return

        # Handle the case when gradient is None (implicit gradient of 1.0)
        if gradient is None:
            if not hasattr(self, '_ones_cache'):
                self._ones_cache = np.ones_like(self.data)
            gradient = self._ones_cache
        elif isinstance(gradient, Tensor):
            gradient = gradient.data

        # Pre-allocate list size for better memory efficiency
        topo = []
        topo.append(self)
        visited = {self}

        # Build topo order iteratively instead of recursively
        idx = 0
        while idx < len(topo):
            current = topo[idx]
            if current._grad_fn is not None:
                for parent in current.parents:
                    if parent not in visited and parent._grad_fn is not None:
                        visited.add(parent)
                        topo.append(parent)
            idx += 1

        # Initialize gradient using in-place operations
        if self.grad is None:
            self.grad = Tensor(gradient)
        else:
            self.grad.data += gradient

        # Backpropagate in reverse topological order
        for tensor in reversed(topo):
            if tensor._grad_fn is not None:
                tensor._grad_fn(tensor.grad)

    @property
    def shape(self):
        """Return the shape of the underlying numpy array"""
        return self.data.shape
    
    def __len__(self) -> int:
        """
        Return the length of the first dimension of the tensor.
        This allows using len() on tensor objects, which is particularly useful
        when working with batches of data or sequences.
        
        Returns:
        int: The size of the first dimension of the tensor
        """
        return self.data.shape[0]
    
    def __getitem__(self, idx) -> 'Tensor':
        """
        Enable tensor indexing and slicing, maintaining autograd functionality.
        Supports integer indexing, slicing, and advanced indexing similar to NumPy.
        
        Parameters:
        idx: Index, slice, or advanced indexing expression
        
        Returns:
        Tensor: A new tensor containing the indexed/sliced data
        """
        result = Tensor(self.data[idx], requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(gradient):
                # Create a gradient array of the same shape as the original data
                grad = np.zeros_like(self.data)
                # Place the gradient in the correct position using the same indexing
                grad[idx] = gradient.data
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __setitem__(self, idx, value):
        """
        Enable setting values through indexing, maintaining autograd functionality.
        
        Parameters:
        idx: Index or slice where to set values
        value: Value(s) to set at the specified indices
        """
        if isinstance(value, Tensor):
            self.data[idx] = value.data
        else:
            self.data[idx] = np.array(value, dtype=self.data.dtype)
    
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
    
    def dot(self, other: 'Tensor') -> 'Tensor':
        """
        Compute the dot product of two tensors.

        Parameters:
        other (Tensor): The tensor to compute the dot product with.

        Returns:
        Tensor: The result of the dot product.
        """
        if not isinstance(other, Tensor):
            raise TypeError("Dot product is only defined between tensors")

        result = (self * other).sum()

        if result.requires_grad:
            def _backward(gradient):
                if self.requires_grad:
                    grad = gradient.data * other.data
                    self.backward(Tensor(grad))
                if other.requires_grad:
                    grad = gradient.data * self.data 
                    other.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def matmul(self, other: 'Tensor') -> 'Tensor':
        """
        Perform matrix multiplication between two tensors.
        Uses __matmul__ which already has backward functionality.

        Parameters:
        other (Tensor): The tensor to multiply with.

        Returns:
        Tensor: The result of the matrix multiplication.
        """
        # @ operator calls __matmul__ which handles gradients
        return self @ other
    
    def add(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Add two tensors element-wise.
        Uses __add__ which already has backward functionality.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to add.

        Returns:
        Tensor: The result of the addition.
        """
        return self + other
    
    def mul(self, other: Union['Tensor', float]) -> 'Tensor':
        """
        Multiply two tensors element-wise.
        Uses __mul__ which already has backward functionality.

        Parameters:
        other (Union['Tensor', float]): The tensor or scalar to multiply.

        Returns:
        Tensor: The result of the multiplication.
        """
        return self * other

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
    
    def cos(self) -> 'Tensor':
        """
        Compute the cosine of each element in the tensor.

        Returns:
        Tensor: The result of the cosine computation.
        """
        result = Tensor(np.cos(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * -np.sin(self.data)
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

    def relu(self) -> 'Tensor':
        """
        Compute the rectified linear unit (ReLU) of each element in the tensor.
        
        Returns:
        Tensor: The result of the ReLU computation.
        """
        result = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * (self.data > 0)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result
    
    def softmax(self, axis=-1) -> 'Tensor':
        """
        Compute the softmax of the tensor along a specified axis.
        
        Parameters:
        axis (int): The axis along which to compute the softmax
        
        Returns:
        Tensor: The result of the softmax computation
        """
        exps = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        result = Tensor(exps / np.sum(exps, axis=axis, keepdims=True), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(gradient):
                sm = result.data
                grad = gradient.data * sm * (1 - sm)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result
    
    def sigmoid(self) -> 'Tensor':
        """
        Compute the sigmoid of each element in the tensor.
        
        Returns:
        Tensor: The result of the sigmoid computation.
        """
        result = Tensor(1 / (1 + np.exp(-self.data)), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            def _backward(gradient):
                grad = gradient.data * result.data * (1 - result.data)
                self.backward(Tensor(grad, requires_grad=self.requires_grad))
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result

    #TODO: add computational graph visualization
    
    def __repr__(self) -> str:
        """Return a string representation of the tensor"""
        return (f"({f"Tensor({self.data}),"
                f"  dtype={self.data.dtype}," 
                f"  requires_grad={self.requires_grad}"})")

    def __str__(self) -> str:
        """Return a string representation of the tensor"""
        return self.__repr__()