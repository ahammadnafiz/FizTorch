# ftensor/utils/math_utils.py
import math
import warnings
from functools import reduce as _reduce
from ftensor.core.base import TensorBase

class Math:
    def safe_divide(self, a, b):
        """
        Safely divide two numbers, returning NaN if division by zero is attempted.
        
        Parameters:
        a (float): Numerator.
        b (float): Denominator.
        
        Returns:
        float: Result of the division or NaN if b is zero.
        """
        if b == 0:
            warnings.warn("Division by zero encountered, returning NaN.", RuntimeWarning)
            return float('nan')
        return a / b

    def sum(self, tensor: TensorBase, axis=None):
        """
        Compute the sum of elements in a tensor along a specified axis.
        
        Parameters:
        tensor (TensorBase): The tensor to sum.
        axis (int, optional): The axis along which to sum. If None, sum all elements.
        
        Returns:
        FTensor: A new tensor containing the sum.
        
        Raises:
        ValueError: If the axis is not 0 or 1.
        """
        if axis is None:
            # Sum all elements in the tensor
            total_sum = self._reduce(tensor.data, lambda x, y: x + y)
            from ftensor import FTensor
            return FTensor([[total_sum]])
        elif axis == 0:
            # Sum along columns
            from ftensor import FTensor
            return FTensor(self._sum_axis_0(tensor.data))
        elif axis == 1:
            # Sum along rows
            from ftensor import FTensor
            return FTensor(self._sum_axis_1(tensor.data))
        else:
            raise ValueError("Invalid axis. Axis must be 0 or 1.")

    def _sum_axis_0(self, data):
        """
        Helper function to sum elements along columns.
        
        Parameters:
        data (list): The data to sum.
        
        Returns:
        list: A list containing the sum of each column.
        """
        return [self._reduce([data[j][i] for j in range(len(data))], lambda x, y: x + y) for i in range(len(data[0]))]

    def _sum_axis_1(self, data):
        """
        Helper function to sum elements along rows.
        
        Parameters:
        data (list): The data to sum.
        
        Returns:
        list: A list containing the sum of each row.
        """
        return [self._reduce(row, lambda x, y: x + y) for row in data]

    def _reduce(self, data, operation):
        """
        Helper function to recursively reduce a nested list using a specified operation.
        
        Parameters:
        data (list): The data to reduce.
        operation (function): The reduction operation (e.g., sum).
        
        Returns:
        The result of the reduction.
        """
        if isinstance(data, list):
            return _reduce(operation, (self._reduce(item, operation) for item in data))
        return data

    def tensor_dot(self, a, b):
        """
        Compute the dot product of two tensors.
        
        Parameters:
        a (list): The first tensor.
        b (list): The second tensor.
        
        Returns:
        The dot product of the two tensors.
        """
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
        """
        Helper function to calculate the tensor product of two matrices.
        
        Parameters:
        a (list): The first matrix.
        b (list): The second matrix.
        
        Returns:
        list: The result of the tensor product.
        """
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                element = self.tensor_dot(a[i], [b_row[j] for b_row in b])
                row.append(element)
            result.append(row)
        return result