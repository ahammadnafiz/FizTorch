import math
import warnings
from functools import reduce as _reduce
from base import TensorBase

class Math:
    def safe_divide(self, a, b):
        if b == 0:
            warnings.warn("Division by zero encountered, returning NaN.", RuntimeWarning)
            return float('nan')
        return a / b

    def sum(self, tensor: TensorBase, axis=None):
        if axis is None:
            total_sum = self._reduce(tensor.data, lambda x, y: x + y)
            from ftensor import FTensor  # Local import
            return FTensor([[total_sum]])
        elif axis == 0:
            from ftensor import FTensor  # Local import
            return FTensor(self._sum_axis_0(tensor.data))
        elif axis == 1:
            from ftensor import FTensor  # Local import
            return FTensor(self._sum_axis_1(tensor.data))
        else:
            raise ValueError("Invalid axis. Axis must be 0 or 1.")

    def _sum_axis_0(self, data):
        return [self._reduce([data[j][i] for j in range(len(data))], lambda x, y: x + y) for i in range(len(data[0]))]

    def _sum_axis_1(self, data):
        return [self._reduce(row, lambda x, y: x + y) for row in data]

    def _reduce(self, data, operation):
        if isinstance(data, list):
            return _reduce(operation, (self._reduce(item, operation) for item in data))
        return data

    def tensor_dot(self, a, b):
        if not isinstance(a, list) or not isinstance(b, list):
            return a * b

        if not a or not b:
            return []

        if not isinstance(a[0], list) and not isinstance(b[0], list):
            return sum(x * y for x, y in zip(a, b))

        if isinstance(a[0], list) and not isinstance(b[0], list):
            return [self.tensor_dot(row, b) for row in a]

        if not isinstance(a[0], list) and isinstance(b[0], list):
            return [sum(a[i] * row[i] for i in range(len(a))) for row in b]

        return self.calculate_tensor_product(a, b)

    def calculate_tensor_product(self, a, b):
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                element = self.tensor_dot(a[i], [b_row[j] for b_row in b])
                row.append(element)
            result.append(row)
        return result