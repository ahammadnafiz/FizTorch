from base import TensorBase

class Operations:
    def elementwise_ops(self, tensor_a: TensorBase, tensor_b: TensorBase, operation):
        self._check_shape(tensor_a, tensor_b)
        from ftensor import FTensor  # Local import
        return FTensor(self._broadcast_and_apply(tensor_a.data, tensor_b.data, operation))
    
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
    
    def _check_shape(self, tensor_a: TensorBase, tensor_b: TensorBase):
        shape_a = tensor_a.shape
        shape_b = tensor_b.shape
        if not self._are_shape_compatible(shape_a, shape_b):
            raise ValueError('Shapes do not match for broadcasting')
    
    def _are_shape_compatible(self, shape_a, shape_b):
        if len(shape_a) != len(shape_b):
            return False
        for dim_a, dim_b in zip(reversed(shape_a), reversed(shape_b)):
            if dim_a != dim_b and dim_a != 1 and dim_b != 1:
                return False
        return True