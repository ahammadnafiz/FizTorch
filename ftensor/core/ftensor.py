# ftensor/core/ftensor.py
from functools import reduce as _reduce
from ftensor.core.base import TensorBase
from ftensor.utils.operations import Operations
from ftensor.utils.shape_utils import Shape
from ftensor.utils.math_utils import Math

import math

class FTensor(TensorBase):
    """
    FTensor class for handling tensor operations.
    This class provides a variety of tensor operations, including element-wise operations, 
    matrix operations, and utility functions for reshaping and manipulating tensor data.
    Attributes:
        _data (list): The underlying data of the tensor.
        operations (Operations): An instance of the Operations class for performing element-wise operations.
        shape_utils (Shape): An instance of the Shape class for handling shape-related utilities.
        math_utils (Math): An instance of the Math class for mathematical operations.
    """
    def __init__(self, data):
        # Initialize the FTensor with data and utility classes
        self._data = data
        self.operations = Operations()
        self.shape_utils = Shape()
        self.math_utils = Math()

    @property
    def data(self):
        # Return the underlying data of the tensor
        return self._data

    @property
    def shape(self):
        # Return the shape of the tensor
        return self.shape_utils.get_shape(self._data)
    
    @property
    def size(self):
        # Return the total number of elements in the tensor
        return _reduce(lambda x, y: x * y, self.shape)
    
    def __add__(self, other):
        # Perform element-wise addition with another tensor
        return self.operations.elementwise_ops(self, other, lambda x, y: x + y)
    
    def __mul__(self, other):
        # Perform element-wise multiplication with another tensor
        return self.operations.elementwise_ops(self, other, lambda x, y: x * y)
    
    def __sub__(self, other):
        # Perform element-wise subtraction with another tensor
        return self.operations.elementwise_ops(self, other, lambda x, y: x - y)

    def __truediv__(self, other):
        # Perform element-wise division with another tensor
        return self.operations.elementwise_ops(self, other, self.math_utils.safe_divide)

    def sum(self, axis=None):
        # Compute the sum of tensor elements along the specified axis
        return self.math_utils.sum(self, axis)

    def dot(self, other):
        # Compute the dot product with another tensor
        return FTensor(self.math_utils.tensor_dot(self._data, other.data))
    
    def transpose(self):
        # Transpose the tensor
        return FTensor(list(map(list, zip(*self._data))))

    def flatten(self):
        # Flatten the tensor into a 1D list
        flat_data = self.shape_utils.flatten(self._data)
        return FTensor(flat_data)
    
    def reshape(self, new_shape):
        # Reshape the tensor to the specified shape
        return self.shape_utils.reshape(self, new_shape)
    
    @property
    def dtype(self):
        # Return the data type of the tensor elements
        return self.shape_utils.get_dtype(self._data)
    
    def __repr__(self):
        # Return a string representation of the tensor
        return self.shape_utils.tensor_repr(self)
    
    def log(self):
        # Apply the natural logarithm element-wise to the tensor
        epsilon = 1e-10
        return self.operations.elementwise_ops(self, self, lambda x, _: float('inf') if x < 0 else math.log(x + epsilon))

    def exp(self):
        # Apply the exponential function element-wise to the tensor
        return self.operations.elementwise_ops(self, self, lambda x, _: math.exp(x))

    def relu(self):
        # Apply the ReLU activation function element-wise to the tensor
        return self.operations.elementwise_ops(self, self, lambda x, _: max(0, x))

    def relu_derivative(self):
        # Compute the derivative of the ReLU function element-wise
        return self.operations.elementwise_ops(self, self, lambda x, _: 1 if x > 0 else 0)

    def sigmoid(self):
        # Apply the sigmoid activation function element-wise to the tensor
        return self.operations.elementwise_ops(self, self, lambda x, _: 1 / (1 + math.exp(-x)))

    def sigmoid_derivative(self):
        # Compute the derivative of the sigmoid function element-wise
        sig = self.sigmoid()
        return self.operations.elementwise_ops(sig, sig, lambda s, _: s * (1 - s))

    def softmax(self):
        # Apply the softmax function to the tensor
        exp_tensor = self.operations.elementwise_ops(self, self, lambda x, _: math.exp(x))
        exp_data = exp_tensor.data if isinstance(exp_tensor, FTensor) else exp_tensor
        flat_exp_data = self.shape_utils.flatten(exp_data)
        sum_exp = sum(flat_exp_data)
        return self.operations.elementwise_ops(exp_tensor, exp_tensor, lambda x, _: x / sum_exp)