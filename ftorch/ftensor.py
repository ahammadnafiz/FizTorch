from functools import reduce as _reduce
from base import TensorBase
from operations import Operations
from shape_utils import Shape
from math_utils import Math

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