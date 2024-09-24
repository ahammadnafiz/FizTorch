from functools import reduce as _reduce
from base import TensorBase

class Shape:
    def get_shape(self, data):
        def _get_shape(data):
            if isinstance(data, list) and data:
                return (len(data),) + _get_shape(data[0])
            return ()
        return _get_shape(data)

    def flatten(self, data):
        def _flatten(data):
            if isinstance(data, list):
                return [item for sublist in data for item in _flatten(sublist)]
            return [data]
        return _flatten(data)

    def reshape(self, tensor: TensorBase, new_shape):
        total_elements = tensor.size
        new_total_elements = _reduce(lambda x, y: x * y, new_shape)

        if total_elements != new_total_elements:
            raise ValueError("Total number of elements must remain the same.")

        flat_data = tensor.flatten().data
        from ftensor import FTensor  # Local import
        return FTensor(self._reshape_helper(flat_data, new_shape))

    def _reshape_helper(self, flat_data, shape):
        if len(shape) == 1:
            return [flat_data[i:i + shape[0]] for i in range(0, len(flat_data), shape[0])]
        
        sub_shape = shape[1:]
        sub_size = _reduce(lambda x, y: x * y, sub_shape)
        return [self._reshape_helper(flat_data[i * sub_size:(i + 1) * sub_size], sub_shape) for i in range(shape[0])]

    def get_dtype(self, data):
        if not data:
            return None
        
        if isinstance(data[0], list):
            return type(data[0][0]).__name__  
        else:
            return type(data[0]).__name__

    def tensor_repr(self, tensor: TensorBase):
        if isinstance(tensor.data[0], list): 
            data_str = '\n'.join(['[' + ' '.join(map(str, row)) + ']' for row in tensor.data[:3]])  # Show only first 3 rows
            if len(tensor.data) > 3:
                data_str += '\n...'
        else:
            data_str = '[' + ' '.join(map(str, tensor.data)) + ']'  
        return f"FTensor(shape={tensor.shape}, dtype={tensor.dtype})\nData:\n{data_str}"