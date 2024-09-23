class MTensor:
    def __init__(self, data):
        self.data = data

    @property
    def shape(self):
        def _get_shape(data):
            if isinstance(data, list) and data:
                return (len(data),) + _get_shape(data[0])
            return ()
        return _get_shape(self.data)
    
    def __add__(self, other):
        return self._elementwise_ops(other, lambda x, y: x + y)
    
    def __mul__(self, other):
        return self._elementwise_ops(other, lambda x, y: x * y)
    
    def __sub__(self, other):
        return self._elementwise_ops(other, lambda x, y: x - y)

    def __truediv__(self, other):
        return self._elementwise_ops(other, lambda x, y: x / y)
    
    def dot(self, other):
        return MTensor(self._tensor_dot(self.data, other.data))
    
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
        return MTensor(self._broadcast_and_apply(self.data, other.data, operation))
    
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
        return type(self.data[0][0]).__name__ if self.data else None
    
    def __repr__(self):
        data_str = '\n'.join(['[' + ' '.join(map(str, row)) + ']' for row in self.data])
        return f"Tensor(shape={self.shape}, dtype={self.dtype})\nData:\n{data_str}"
    
a = MTensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
b = MTensor([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])
result = a.dot(b)
print(a + b)
print(a - b)
print(a * b)
# print(a / b) # have to work with divisionbyzero error
print(result)