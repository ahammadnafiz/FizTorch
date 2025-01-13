import numpy as np

class GradientUtils:
    """
    A utility class for gradient-related operations.
    """

    @staticmethod
    def unbroadcast(grad, shape):
        """
        Unbroadcast gradients to match the original tensor shape.

        Parameters:
        grad (np.ndarray): The gradient array to be unbroadcasted.
        shape (tuple): The target shape to unbroadcast to.

        Returns:
        np.ndarray: The unbroadcasted gradient.
        """
        # Handle scalars
        if not shape:
            return np.sum(grad)

        # Sum out the broadcasted dimensions
        axes = tuple(range(len(grad.shape) - len(shape)))  # Leading dimensions
        for i, (grad_size, shape_size) in enumerate(zip(grad.shape[len(axes):], shape)):
            if grad_size != shape_size:
                axes += (i + len(axes),)
        
        if axes:
            return np.sum(grad, axis=axes).reshape(shape)
        
        return grad