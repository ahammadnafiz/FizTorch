from typing import Union, Optional, Set, List, Dict
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
            self.data = data.astype(np.float64)
        else:
            self.data = np.array(data, dtype=np.float64)

        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        self._grad_fn = None
        self.is_leaf = True
        self._backward_refs = set()  # Keep track of tensors that depend on this one
        self._cached_grad = None     # For gradient caching

    def _add_backward_ref(self, tensor: 'Tensor') -> None:
        """Add a reference to a tensor that depends on this one."""
        if self.requires_grad:
            self._backward_refs.add(tensor)

    @staticmethod
    def _topological_sort(tensor: 'Tensor') -> List['Tensor']:
        """
        Perform topological sort on the computation graph starting from the given tensor.
        Returns a list of tensors in reverse order of computation (for backward pass).
        """
        visited = set()
        topo_order = []

        def visit(t: 'Tensor'):
            if t not in visited and t.requires_grad:
                visited.add(t)
                for dep in t._backward_refs:
                    visit(dep)
                topo_order.append(t)

        visit(tensor)
        return topo_order

    def backward(self, gradient: Optional[Union['Tensor', np.ndarray]] = None) -> None:
        """
        Compute gradients using an iterative approach with topological sorting.
        
        Parameters:
        gradient (Optional[Union['Tensor', np.ndarray]]): The gradient to be propagated.
        """
        if not self.requires_grad:
            return

        # Initialize gradient if None
        if gradient is None:
            gradient = np.ones_like(self.data)
        elif isinstance(gradient, Tensor):
            gradient = gradient.data

        # Get computation graph in reverse order
        topo_order = self._topological_sort(self)
        
        # Initialize gradients dictionary
        grads = {self: gradient}
        
        # Iterate through the computation graph in reverse order
        for tensor in topo_order:
            grad = grads.pop(tensor, None)
            
            # Skip if no gradient is flowing through this node
            if grad is None:
                continue

            # For leaf tensors, accumulate the gradient
            if tensor.is_leaf:
                if tensor.grad is None:
                    tensor.grad = Tensor(grad)
                else:
                    tensor.grad.data += grad
                continue

            # If we have cached gradients and the tensor hasn't changed, use them
            if tensor._cached_grad is not None and tensor._grad_fn is not None:
                cached_grads = tensor._cached_grad
                for t, g in cached_grads.items():
                    if t in grads:
                        grads[t] += g * grad
                    else:
                        grads[t] = g * grad
                continue

            # Compute gradients using the gradient function
            if tensor._grad_fn is not None:
                local_grads = {}
                
                def accumulate_grad(t: 'Tensor', g: np.ndarray):
                    if t in local_grads:
                        local_grads[t] += g
                    else:
                        local_grads[t] = g

                # Modified gradient function that accumulates gradients
                def wrapped_backward(gradient_tensor):
                    if isinstance(gradient_tensor, Tensor):
                        g_data = gradient_tensor.data
                    else:
                        g_data = gradient_tensor
                    accumulate_grad(self, g_data)

                # Call the original gradient function
                tensor._grad_fn(Tensor(grad))
                
                # Cache the computed gradients
                tensor._cached_grad = local_grads

                # Accumulate gradients in the global gradients dictionary
                for t, g in local_grads.items():
                    if t in grads:
                        grads[t] += g
                    else:
                        grads[t] = g

    def zero_grad(self, clear_cache: bool = True):
        """
        Reset the gradient to zero and optionally clear gradient cache.
        
        Parameters:
        clear_cache (bool): If True, clear the cached gradients
        """
        if self.requires_grad:
            self.grad = None
            if clear_cache:
                self._cached_grad = None

    def clear_graph(self):
        """Clear the computation graph to free memory."""
        self._backward_refs.clear()
        self._cached_grad = None
        self._grad_fn = None

    @property
    def shape(self):
        """Return the shape of the underlying numpy array"""
        return self.data.shape
    
    def __len__(self) -> int:
        """Return the length of the first dimension of the tensor."""
        return self.data.shape[0]
    
    def __getitem__(self, idx) -> 'Tensor':
        """Enable tensor indexing and slicing."""
        result = Tensor(self.data[idx], requires_grad=self.requires_grad)
        
        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = np.zeros_like(self.data)
                np.add.at(grad, idx, gradient.data)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __setitem__(self, idx, value):
        """Enable setting values through indexing."""
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
            self._add_backward_ref(result)
            def _backward(gradient):
                self.backward(Tensor(gradient.data.T))
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result

    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Add two tensors element-wise."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        result = Tensor(self.data + other_data, requires_grad=self.requires_grad or 
                    (isinstance(other, Tensor) and other.requires_grad))

        if result.requires_grad:
            if self.requires_grad:
                self._add_backward_ref(result)
            if isinstance(other, Tensor) and other.requires_grad:
                other._add_backward_ref(result)

            def _backward(gradient):
                if self.requires_grad:
                    unbroadcast_grad = _GradientUtils.unbroadcast(gradient.data, self.data.shape)
                    self.backward(Tensor(unbroadcast_grad))
                if isinstance(other, Tensor) and other.requires_grad:
                    unbroadcast_grad = _GradientUtils.unbroadcast(gradient.data, other.data.shape)
                    other.backward(Tensor(unbroadcast_grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Multiply two tensors element-wise."""
        other_data = other.data if isinstance(other, Tensor) else np.array(other, dtype=np.float64)
        result = Tensor(self.data * other_data, requires_grad=self.requires_grad or 
                    (isinstance(other, Tensor) and other.requires_grad))

        if result.requires_grad:
            if self.requires_grad:
                self._add_backward_ref(result)
            if isinstance(other, Tensor) and other.requires_grad:
                other._add_backward_ref(result)

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
        """Perform matrix multiplication."""
        if not isinstance(other, Tensor):
            raise TypeError("Matrix multiplication is only defined between tensors")
            
        result = Tensor(self.data @ other.data, 
                       requires_grad=(self.requires_grad or other.requires_grad))

        if result.requires_grad:
            if self.requires_grad:
                self._add_backward_ref(result)
            if other.requires_grad:
                other._add_backward_ref(result)

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
        """Subtract two tensors element-wise."""
        return self + (-other if isinstance(other, Tensor) else -np.array(other))

    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Subtract this tensor from another tensor or scalar."""
        return (-self) + (other if isinstance(other, Tensor) else np.array(other))

    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        """Divide this tensor by another tensor or scalar element-wise."""
        if isinstance(other, (int, float, np.integer)):
            return self * (1.0 / float(other))
        return self * (other ** -1)

    def __pow__(self, power: float) -> 'Tensor':
        """Raise the tensor to a power element-wise."""
        result = Tensor(self.data ** power, requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * (power * self.data ** (power - 1))
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        """Compute the sum of tensor elements over given axes."""
        result = Tensor(np.sum(self.data, axis=axis, keepdims=keepdims), 
                       requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
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
        """Compute the mean of tensor elements over given axes."""
        return self.sum(axis=axis, keepdims=keepdims) / np.prod(np.array(self.data.shape)[axis] if axis is not None else self.data.shape)

    def reshape(self, *shape) -> 'Tensor':
        """Reshape the tensor to a new shape."""
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            shape = shape[0]
        
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                self.backward(Tensor(gradient.data.reshape(self.data.shape)))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def exp(self) -> 'Tensor':
        """Compute the exponential of each element."""
        result = Tensor(np.exp(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * np.exp(self.data)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def log(self) -> 'Tensor':
        """Compute the natural logarithm of each element."""
        result = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        
        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data / self.data
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False
        
        return result

    def sin(self) -> 'Tensor':
        """Compute the sine of each element."""
        result = Tensor(np.sin(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * np.cos(self.data)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result
    
    def clip_grad_(self, min_val: float = None, max_val: float = None) -> None:
        """Clip gradients to a specified range in-place."""
        if self.grad is not None:
            if min_val is not None:
                self.grad.data = np.maximum(self.grad.data, min_val)
            if max_val is not None:
                self.grad.data = np.minimum(self.grad.data, max_val)

    def cos(self) -> 'Tensor':
        """Compute the cosine of each element."""
        result = Tensor(np.cos(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * -np.sin(self.data)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def tan(self) -> 'Tensor':
        """Compute the tangent of each element."""
        result = Tensor(np.tan(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * (1 + np.tan(self.data) ** 2)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def tanh(self) -> 'Tensor':
        """Compute the hyperbolic tangent of each element."""
        result = Tensor(np.tanh(self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * (1 - np.tanh(self.data) ** 2)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def relu(self) -> 'Tensor':
        """Apply the ReLU activation function."""
        result = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * (self.data > 0)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def sigmoid(self) -> 'Tensor':
        """Apply the sigmoid activation function."""
        sig = 1 / (1 + np.exp(-self.data))
        result = Tensor(sig, requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = gradient.data * sig * (1 - sig)
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def softmax(self, axis=-1) -> 'Tensor':
        """Apply the softmax function along the specified axis."""
        exp_data = np.exp(self.data - np.max(self.data, axis=axis, keepdims=True))
        softmax_data = exp_data / np.sum(exp_data, axis=axis, keepdims=True)
        result = Tensor(softmax_data, requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                s = softmax_data
                grad = s * (gradient.data - np.sum(gradient.data * s, axis=axis, keepdims=True))
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def max(self, axis=None, keepdims=False) -> 'Tensor':
        """Compute the maximum value along the specified axis."""
        result = Tensor(np.max(self.data, axis=axis, keepdims=keepdims),
                       requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = np.zeros_like(self.data)
                if axis is not None:
                    idx = list(np.indices(self.data.shape))
                    max_idx = list(self.data.argmax(axis=axis))
                    if not keepdims:
                        for i in range(axis, len(idx)):
                            max_idx.insert(i, slice(None))
                    grad[tuple(max_idx)] = gradient.data
                else:
                    grad[np.unravel_index(self.data.argmax(), self.data.shape)] = gradient.data
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def min(self, axis=None, keepdims=False) -> 'Tensor':
        """Compute the minimum value along the specified axis."""
        result = Tensor(np.min(self.data, axis=axis, keepdims=keepdims),
                       requires_grad=self.requires_grad)

        if self.requires_grad:
            self._add_backward_ref(result)
            def _backward(gradient):
                grad = np.zeros_like(self.data)
                if axis is not None:
                    idx = list(np.indices(self.data.shape))
                    min_idx = list(self.data.argmin(axis=axis))
                    if not keepdims:
                        for i in range(axis, len(idx)):
                            min_idx.insert(i, slice(None))
                    grad[tuple(min_idx)] = gradient.data
                else:
                    grad[np.unravel_index(self.data.argmin(), self.data.shape)] = gradient.data
                self.backward(Tensor(grad))
            result._grad_fn = _backward
            result.is_leaf = False

        return result

    def detach(self) -> 'Tensor':
        """Create a new Tensor with the same data but detached from the computation graph."""
        return Tensor(self.data, requires_grad=False)

    def clone(self) -> 'Tensor':
        """Create a new Tensor with the same data and gradient requirements."""
        return Tensor(self.data.copy(), requires_grad=self.requires_grad)

    def __repr__(self) -> str:
        """Return a string representation of the tensor."""
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __str__(self) -> str:
        """Return a string representation of the tensor."""
        return self.__repr__()