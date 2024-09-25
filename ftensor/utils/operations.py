# ftensor/utils/operations.py
from ftensor.core.base import TensorBase

class Operations:
    '''The Operations class provides methods for performing element-wise operations on tensors with support for broadcasting.
    Methods:
        elementwise_ops(tensor_a, tensor_b, operation):
        _broadcast_and_apply(a, b, operation, shape_a, shape_b):
        _get_broadcast_shape(shape_a, shape_b):
        _broadcast(data, from_shape, to_shape):
        _apply_operation(a, b, operation):
        _are_shape_compatible(shape_a, shape_b):'''
        
    def elementwise_ops(self, tensor_a: 'TensorBase', tensor_b: 'TensorBase', operation):
        """
        Perform element-wise operations on two tensors with broadcasting support.

        Parameters:
        tensor_a (TensorBase): The first input tensor.
        tensor_b (TensorBase): The second input tensor.
        operation (callable): A function that takes two arguments and performs an element-wise operation.

        Returns:
        FTensor: A new tensor resulting from the element-wise operation.
        """
        if not self._are_shape_compatible(tensor_a.shape, tensor_b.shape):
            raise ValueError(f'Shapes do not match for broadcasting: {tensor_a.shape} and {tensor_b.shape}')
        from ftensor import FTensor
        result_data = self._broadcast_and_apply(tensor_a.data, tensor_b.data, operation, tensor_a.shape, tensor_b.shape)
        return FTensor(result_data)

    def _broadcast_and_apply(self, a, b, operation, shape_a, shape_b):
        """
        Broadcasts two input arrays to a common shape and applies a specified operation element-wise.

        Parameters:
        a (array-like): The first input array.
        b (array-like): The second input array.
        operation (callable): A function that takes two arguments and performs an element-wise operation.
        shape_a (tuple): The shape of the first input array.
        shape_b (tuple): The shape of the second input array.

        Returns:
        array-like: The result of applying the operation to the broadcasted arrays.
        """
        # Calculate the broadcasted shape for the input shapes
        output_shape = self._get_broadcast_shape(shape_a, shape_b)
        
        # Broadcast the input data 'a' and 'b' to the calculated output shape
        a_broadcast = self._broadcast(a, shape_a, output_shape)
        b_broadcast = self._broadcast(b, shape_b, output_shape)
        
        # Apply the given operation element-wise to the broadcasted data
        return self._apply_operation(a_broadcast, b_broadcast, operation)

    def _get_broadcast_shape(self, shape_a, shape_b):
        """
        Calculate the broadcasted shape for two input shapes.

        This method takes two shapes (tuples of integers) and computes the 
        broadcasted shape according to the broadcasting rules of numpy. 
        Broadcasting is the process of making arrays with different shapes 
        have compatible shapes for element-wise operations.

        Parameters:
        shape_a (tuple): The shape of the first array.
        shape_b (tuple): The shape of the second array.

        Returns:
        tuple: The broadcasted shape.

        Example:
        >>> _get_broadcast_shape((2, 3), (3,))
        (2, 3)
        >>> _get_broadcast_shape((1, 2, 3), (3,))
        (1, 2, 3)
        """
        # Determine the maximum number of dimensions between the two shapes
        max_dims = max(len(shape_a), len(shape_b))
        
        # Pad the shorter shape with ones on the left to match the number of dimensions
        shape_a = (1,) * (max_dims - len(shape_a)) + shape_a
        shape_b = (1,) * (max_dims - len(shape_b)) + shape_b
        
        # Compute the broadcasted shape by taking the maximum value along each dimension
        return tuple(max(sa, sb) for sa, sb in zip(shape_a, shape_b))

    def _broadcast(self, data, from_shape, to_shape):
        """
        Broadcast the input data from `from_shape` to `to_shape`.

        This implementation handles higher-dimensional broadcasting by ensuring
        singleton dimensions (1) are expanded as per the broadcasting rules.

        Parameters:
        data (array-like): The input data to be broadcasted.
        from_shape (tuple): The original shape of the input data.
        to_shape (tuple): The target shape to broadcast the data to.

        Returns:
        array-like: The broadcasted data.
        """
        # If the shapes already match, return the data as-is
        if from_shape == to_shape:
            return data
        
        # Initialize the result with the input data
        result = data
        
        # Iterate through each dimension starting from the last (rightmost)
        for i in range(len(to_shape) - 1, -1, -1):
            if i >= len(from_shape):  # Broadcast missing dimensions in `from_shape`
                result = [result] * to_shape[i]
            elif from_shape[i] == 1:  # Broadcast singleton dimensions (1)
                result = [result for _ in range(to_shape[i])]
            elif from_shape[i] != to_shape[i]:  # Incompatible dimensions
                raise ValueError(f"Cannot broadcast shape {from_shape} to {to_shape}")

        return result

    def _apply_operation(self, a, b, operation):
        """
        Recursively apply an element-wise operation to two broadcasted arrays.

        This method applies the given operation to each element of the two 
        broadcasted arrays `a` and `b`. It handles nested lists (representing 
        higher-dimensional arrays) by recursively applying the operation to 
        corresponding elements.

        Parameters:
        a (list or scalar): The first broadcasted array or scalar.
        b (list or scalar): The second broadcasted array or scalar.
        operation (callable): A function that takes two arguments and returns 
                              the result of the element-wise operation.

        Returns:
        list or scalar: The result of applying the operation to each element 
                        of the broadcasted arrays.
        """
        if isinstance(a, list) and isinstance(b, list):
            # Both `a` and `b` are lists, apply the operation element-wise
            return [self._apply_operation(ai, bi, operation) for ai, bi in zip(a, b)]
        elif isinstance(a, list):
            # Only `a` is a list, apply the operation between each element of `a` and `b`
            return [self._apply_operation(ai, b, operation) for ai in a]
        elif isinstance(b, list):
            # Only `b` is a list, apply the operation between `a` and each element of `b`
            return [self._apply_operation(a, bi, operation) for bi in b]
        # Both `a` and `b` are scalars, apply the operation directly
        return operation(a, b)

    def _are_shape_compatible(self, shape_a, shape_b):
        """
        Check if two shapes are compatible for broadcasting.

        This method determines whether two shapes can be broadcast together
        according to the broadcasting rules. Broadcasting is the process of 
        making arrays with different shapes have compatible shapes for element-wise 
        operations.

        Parameters:
        shape_a (tuple): The shape of the first array.
        shape_b (tuple): The shape of the second array.

        Returns:
        bool: True if the shapes are compatible for broadcasting, False otherwise.
        """
        try:
            # Attempt to get the broadcast shape for the given shapes
            # If the shapes are not compatible, this will raise a ValueError
            self._get_broadcast_shape(shape_a, shape_b)
            return True
        except ValueError:
            # If a ValueError is raised, the shapes are not compatible for broadcasting
            return False