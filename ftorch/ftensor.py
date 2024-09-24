from functools import reduce as _reduce
from base import TensorBase
from operations import Operations
from shape_utils import Shape
from math_utils import Math

import math

class FTensor(TensorBase):
    def __init__(self, data):
        self._data = data
        self.operations = Operations()
        self.shape_utils = Shape()
        self.math_utils = Math()

    @property
    def data(self):
        return self._data

    @property
    def shape(self):
        return self.shape_utils.get_shape(self._data)
    
    @property
    def size(self):
        return _reduce(lambda x, y: x * y, self.shape)
    
    def __add__(self, other):
        return self.operations.elementwise_ops(self, other, lambda x, y: x + y)
    
    def __mul__(self, other):
        return self.operations.elementwise_ops(self, other, lambda x, y: x * y)
    
    def __sub__(self, other):
        return self.operations.elementwise_ops(self, other, lambda x, y: x - y)

    def __truediv__(self, other):
        return self.operations.elementwise_ops(self, other, self.math_utils.safe_divide)

    def sum(self, axis=None):
        return self.math_utils.sum(self, axis)

    def dot(self, other):
        return FTensor(self.math_utils.tensor_dot(self._data, other.data))
    
    def transpose(self):
        return FTensor(list(map(list, zip(*self._data))))

    def flatten(self):
        flat_data = self.shape_utils.flatten(self._data)
        return FTensor(flat_data)
    
    def reshape(self, new_shape):
        return self.shape_utils.reshape(self, new_shape)
    
    @property
    def dtype(self):
        return self.shape_utils.get_dtype(self._data)
    
    def __repr__(self):
        return self.shape_utils.tensor_repr(self)
    
    def log(self):
        epsilon = 1e-10  # Small number to prevent log(0)

        def safe_log(x, _):
            # Return positive infinity for negative inputs, log(x + epsilon) for non-negative inputs
            return float('inf') if x < 0 else math.log(x + epsilon)

        return self.operations.elementwise_ops(self, self, safe_log)

    def exp(self):
        return self.operations.elementwise_ops(self, self, lambda x, _: math.exp(x))

    def relu(self):
        return self.operations.elementwise_ops(self, self, lambda x, _: max(0, x))

    def relu_derivative(self):
        return self.operations.elementwise_ops(self, self, lambda x, _: 1 if x > 0 else 0)

    def sigmoid(self):
        return self.operations.elementwise_ops(self, self, lambda x, _: 1 / (1 + math.exp(-x)))

    def sigmoid_derivative(self):
        sig = self.sigmoid(self)
        return self.operations.elementwise_ops(sig, sig, lambda s, _: s * (1 - s))

    def softmax(self):
        # Apply the exponential function element-wise
        exp_tensor = self.operations.elementwise_ops(self, self, lambda x, _: math.exp(x))

        # Ensure exp_tensor is iterable (if it's a tensor, extract the data)
        if isinstance(exp_tensor, FTensor):
            exp_data = exp_tensor.data
        else:
            exp_data = exp_tensor  # If it's not an FTensor, just use it directly

        # Compute the sum of exponentials using flatten
        flat_exp_data = self.shape_utils.flatten(exp_data)
        sum_exp = sum(flat_exp_data)  # This will now work correctly

        # Calculate the softmax values
        return self.operations.elementwise_ops(exp_tensor, exp_tensor, lambda x, _: x / sum_exp)