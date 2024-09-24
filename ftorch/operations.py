from base import TensorBase

class Operations:
    def elementwise_ops(self, tensor_a: 'TensorBase', tensor_b: 'TensorBase', operation):
        if not self._are_shape_compatible(tensor_a.shape, tensor_b.shape):
            raise ValueError(f'Shapes do not match for broadcasting: {tensor_a.shape} and {tensor_b.shape}')
        from ftensor import FTensor  # Local import
        result_data = self._broadcast_and_apply(tensor_a.data, tensor_b.data, operation, tensor_a.shape, tensor_b.shape)
        return FTensor(result_data)

    def _broadcast_and_apply(self, a, b, operation, shape_a, shape_b):
        output_shape = self._get_broadcast_shape(shape_a, shape_b)
        a_broadcast = self._broadcast(a, shape_a, output_shape)
        b_broadcast = self._broadcast(b, shape_b, output_shape)
        return self._apply_operation(a_broadcast, b_broadcast, operation)

    def _get_broadcast_shape(self, shape_a, shape_b):
        max_dims = max(len(shape_a), len(shape_b))
        shape_a = (1,) * (max_dims - len(shape_a)) + shape_a
        shape_b = (1,) * (max_dims - len(shape_b)) + shape_b
        return tuple(max(sa, sb) for sa, sb in zip(shape_a, shape_b))

    def _broadcast(self, data, from_shape, to_shape):
        if from_shape == to_shape:
            return data
        
        result = data
        for i in range(len(to_shape) - 1, -1, -1):
            if i >= len(from_shape) or from_shape[i] == 1:
                result = [result] * to_shape[i]
            elif from_shape[i] != to_shape[i]:
                raise ValueError(f"Cannot broadcast shape {from_shape} to {to_shape}")
        
        return result

    def _apply_operation(self, a, b, operation):
        if isinstance(a, list) and isinstance(b, list):
            return [self._apply_operation(ai, bi, operation) for ai, bi in zip(a, b)]
        elif isinstance(a, list):
            return [self._apply_operation(ai, b, operation) for ai in a]
        elif isinstance(b, list):
            return [self._apply_operation(a, bi, operation) for bi in b]
        return operation(a, b)

    def _are_shape_compatible(self, shape_a, shape_b):
        try:
            self._get_broadcast_shape(shape_a, shape_b)
            return True
        except ValueError:
            return False