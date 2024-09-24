from functools import reduce as _reduce
import warnings

class FTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        def _get_shape(data):
            if isinstance(data, list) and data:
                return (len(data),) + _get_shape(data[0])
            return ()
        return _get_shape(self.data)
    
    @property
    def size(self):
        return _reduce(lambda x, y: x * y, self.shape)
    
    def __add__(self, other):
        return self._elementwise_ops(other, lambda x, y: x + y)
    
    def __mul__(self, other):
        return self._elementwise_ops(other, lambda x, y: x * y)
    
    def __sub__(self, other):
        return self._elementwise_ops(other, lambda x, y: x - y)

    def __truediv__(self, other):
        return self._elementwise_ops(other, self._safe_divide)

    def _safe_divide(self, a, b):
        if b == 0:
            warnings.warn("Division by zero encountered, returning NaN.", RuntimeWarning)
            return float('nan')
        return a / b

    def sum(self, axis=None):
        if axis is None:
            # Calculate the overall sum and wrap it in an FTensor
            total_sum = self._reduce(self.data, lambda x, y: x + y)
            return FTensor([[total_sum]])  # Return as a 1x1 tensor
        elif axis == 0:
            return FTensor(self._sum_axis_0(self.data))
        elif axis == 1:
            return FTensor(self._sum_axis_1(self.data))
        else:
            raise ValueError("Invalid axis. Axis must be 0 or 1.")

    def _sum_axis_0(self, data):
        return [self._reduce([data[j][i] for j in range(len(data))], lambda x, y: x + y) for i in range(len(data[0]))]

    def _sum_axis_1(self, data):
        return [self._reduce(row, lambda x, y: x + y) for row in data]
    
    def dot(self, other):
        return FTensor(self._tensor_dot(self.data, other.data))
    
    def transpose(self):
        return FTensor(list(map(list, zip(*self.data))))

    def flatten(self):
        def _flatten(data):
            if isinstance(data, list):
                return [item for sublist in data for item in _flatten(sublist)]
            return [data]  # Return as a list for non-list items

        flat_data = _flatten(self.data)
        return FTensor(flat_data)

    def _reduce(self, data, operation):
        if isinstance(data, list):
            return _reduce(operation, (self._reduce(item, operation) for item in data))
        return data
    
    def reshape(self, new_shape):
        total_elements = self.size()
        new_total_elements = _reduce(lambda x, y: x * y, new_shape)

        if total_elements != new_total_elements:
            raise ValueError("Total number of elements must remain the same.")

        flat_data = self.flatten().data
        return FTensor(self._reshape_helper(flat_data, new_shape))

    def _reshape_helper(self, flat_data, shape):
        if len(shape) == 1:
            return [flat_data[i:i + shape[0]] for i in range(0, len(flat_data), shape[0])]
        
        sub_shape = shape[1:]
        sub_size = _reduce(lambda x, y: x * y, sub_shape)
        return [self._reshape_helper(flat_data[i * sub_size:(i + 1) * sub_size], sub_shape) for i in range(shape[0])]
    
    def _tensor_dot(self, a, b):
        if not isinstance(a, list) or not isinstance(b, list):
            return a * b

        if not a or not b:
            return []

        if not isinstance(a[0], list) and not isinstance(b[0], list):
            return sum(x * y for x, y in zip(a, b))

        if isinstance(a[0], list) and not isinstance(b[0], list):
            return [self._tensor_dot(row, b) for row in a]

        if not isinstance(a[0], list) and isinstance(b[0], list):
            return [sum(a[i] * row[i] for i in range(len(a))) for row in b]

        # For higher dimensions
        return self.calculate_tensor_product(a, b)

    def calculate_tensor_product(self, a, b):
        result = []
        for i in range(len(a)):
            row = []
            for j in range(len(b[0])):
                element = self._tensor_dot(a[i], [b_row[j] for b_row in b])
                row.append(element)
            result.append(row)
        return result

    def _initialize_tensor(self, shape):
        if len(shape) == 1:
            return [0] * shape[0]
        return [self._initialize_tensor(shape[1:]) for _ in range(shape[0])]
    
    def _elementwise_ops(self, other, operation):
        self._check_shape(other)
        return FTensor(self._broadcast_and_apply(self.data, other.data, operation))
    
    def _broadcast_and_apply(self, a, b, operation):
        if isinstance(a, list) and isinstance(b, list):
            if len(a) == len(b):
                return [self._broadcast_and_apply(a[i], b[i], operation) for i in range(len(a))]
            elif len(a) == 1:
                return [self._broadcast_and_apply(a[0], b[i], operation) for i in range(len(b))]
            elif len(b) == 1:
                return [self._broadcast_and_apply(a[i], b[0], operation) for i in range(len(a))]
        elif isinstance(a, list):
            return [self._broadcast_and_apply(a[i], b, operation) for i in range(len(a))]
        elif isinstance(b, list):
            return [self._broadcast_and_apply(a, b[i], operation) for i in range(len(b))]
        else:
            return operation(a, b)
    
    def _check_shape(self, other):
        shape_a = self.shape
        shape_b = other.shape
        if not self._are_shape_compatible(shape_a, shape_b):
            raise ValueError('Shapes do not match for broadcasting')
    
    def _are_shape_compatible(self, shape_a, shape_b):
        if len(shape_a) != len(shape_b):
            return False
        for dim_a, dim_b in zip(reversed(shape_a), reversed(shape_b)):
            if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                return False
        return True
    
    @property
    def dtype(self):
        if not self.data:
            return None
        
        if isinstance(self.data[0], list):
            return type(self.data[0][0]).__name__  
        else:
            return type(self.data[0]).__name__  
    
    def __repr__(self):
        if isinstance(self.data[0], list): 
            data_str = '\n'.join(['[' + ' '.join(map(str, row)) + ']' for row in self.data[:3]])  # Show only first 3 rows
            if len(self.data) > 3:
                data_str += '\n...'
        else:
            data_str = '[' + ' '.join(map(str, self.data)) + ']'  
        return f"FTensor(shape={self.shape}, dtype={self.dtype})\nData:\n{data_str}"


a = FTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = FTensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
result = a.dot(b)
print(a + b)
print(a - b)
print(a * b)
# print(a / b)
print(result)
print(a.flatten())
print(b.transpose())
c = FTensor([
    [1,2, 4]
])
print(c.shape)
print(c.transpose().shape)
d = FTensor([[1, 2, 3], [4, 5, 6]])
print(d.sum(axis=0))
print(d.sum(axis=1))
print(d.sum())
print(b.size)