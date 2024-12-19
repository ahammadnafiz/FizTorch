import numpy as np
from ..tensor import Tensor

def uniform_(tensor: Tensor, a: float = 0., b: float = 1.) -> Tensor:
    """Fills the tensor with values drawn from the uniform distribution U(a, b)."""
    tensor.data = np.random.uniform(a, b, size=tensor.shape)
    return tensor

def normal_(tensor: Tensor, mean: float = 0., std: float = 1.) -> Tensor:
    """Fills the tensor with values drawn from the normal distribution N(mean, std²)."""
    tensor.data = np.random.normal(mean, std, size=tensor.shape)
    return tensor

def xavier_uniform_(tensor: Tensor) -> Tensor:
    """Fills the tensor with values using Xavier uniform initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    bound = np.sqrt(6. / (fan_in + fan_out))
    return uniform_(tensor, -bound, bound)

def xavier_normal_(tensor: Tensor) -> Tensor:
    """Fills the tensor with values using Xavier normal initialization."""
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = np.sqrt(2. / (fan_in + fan_out))
    return normal_(tensor, 0., std)

def zeros_(tensor: Tensor) -> Tensor:
    """Fills the tensor with zeros."""
    tensor.data = np.zeros_like(tensor.data)
    return tensor

def ones_(tensor: Tensor) -> Tensor:
    """Fills the tensor with ones."""
    tensor.data = np.ones_like(tensor.data)
    return tensor

def _calculate_fan_in_and_fan_out(tensor: Tensor) -> tuple:
    """Computes the number of input and output features for a tensor."""
    dimensions = tensor.shape
    if len(dimensions) == 2:  # Linear
        fan_in, fan_out = dimensions
    else:
        raise ValueError("Only 2D tensors supported for now")
    return fan_in, fan_out