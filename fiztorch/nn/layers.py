from typing import Union, Tuple
import numpy as np
from ..tensor import Tensor
from .module import Module
from .init_functions import xavier_uniform_, kaiming_uniform_, zeros_
from .activations import (
    ReLU as ReLUActivation,
    LeakyReLU as LeakyReLUActivation,
    ELU as ELUActivation,
    SELU as SELUActivation,
    Sigmoid as SigmoidActivation,
    Softmax as SoftmaxActivation,
    Tanh as TanhActivation
)

class Linear(Module):
    """A linear transformation layer (fully connected layer).
    
    Applies a linear transformation to the incoming data: y = xA^T + b
    
    Args:
        in_features: Size of each input sample
        out_features: Size of each output sample
        bias: If set to False, the layer will not learn an additive bias. Default: True
        init_method: Initialization method for weights. One of ['xavier', 'kaiming']. Default: 'xavier'
        
    Shape:
        - Input: (*, in_features) where * means any number of additional dimensions
        - Output: (*, out_features)
        
    Examples:
        >>> m = Linear(20, 30)
        >>> input = Tensor(np.random.randn(128, 20))
        >>> output = m(input)
        >>> print(output.shape)
        (128, 30)
    """
    
    def __init__(
        self, 
        in_features: int,
        out_features: int,
        bias: bool = True,
        init_method: str = 'xavier'
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        
        # Initialize weights
        self.weight = Tensor(
            np.empty((out_features, in_features)),
            requires_grad=True
        )
        
        if init_method == 'xavier':
            xavier_uniform_(self.weight)
        elif init_method == 'kaiming':
            kaiming_uniform_(self.weight)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")
            
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = Tensor(np.empty(out_features), requires_grad=True)
            zeros_(self.bias)
            self._parameters['bias'] = self.bias
        else:
            self.register_parameter('bias', None)
            
    def forward(self, x: Tensor) -> Tensor:
        """Forward pass of the layer.
        
        Args:
            x: Input tensor of shape (*, in_features)
            
        Returns:
            Output tensor of shape (*, out_features)
        """
        if not isinstance(x, Tensor):
            x = Tensor(x, requires_grad=True)
            
        # Ensure parameters are properly tracked
        self._ensure_parameters_tracking()
        
        # Linear transformation
        out = x @ self.weight.T
        if self.bias is not None:
            out = out + self.bias
            
        # Define backward pass
        if x.requires_grad:
            def _backward(gradient: Tensor) -> None:
                if self.weight.requires_grad:
                    weight_grad = gradient.data.T @ x.data
                    self.weight.backward(Tensor(weight_grad, requires_grad=True))
                if self.bias is not None and self.bias.requires_grad:
                    bias_grad = gradient.data.sum(axis=0)
                    self.bias.backward(Tensor(bias_grad, requires_grad=True))
                    
            out._grad_fn = _backward
            out.is_leaf = False
            
        return out
    
    def _ensure_parameters_tracking(self) -> None:
        """Ensures all parameters are properly set for gradient tracking."""
        if not self.weight.requires_grad:
            self.weight = Tensor(self.weight.data, requires_grad=True)
        if self.bias is not None and not self.bias.requires_grad:
            self.bias = Tensor(self.bias.data, requires_grad=True)
    
    def extra_repr(self) -> str:
        """Extra representation of the layer when printed."""
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'


class Conv2d(Module):
    """2D convolution layer.
    
    Args:
        in_channels: Number of input channels
        out_channels: Number of output channels
        kernel_size: Size of the convolving kernel
        stride: Stride of the convolution. Default: 1
        padding: Padding added to all sides of the input. Default: 0
        bias: If True, adds a learnable bias to the output. Default: True
    """
    
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        bias: bool = True
    ) -> None:
        super().__init__()
        if isinstance(kernel_size, int):
            kernel_size = (kernel_size, kernel_size)
        if isinstance(stride, int):
            stride = (stride, stride)
        if isinstance(padding, int):
            padding = (padding, padding)
            
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights
        self.weight = Tensor(
            np.random.randn(out_channels, in_channels, *kernel_size) * 0.01,
            requires_grad=True
        )
        self._parameters['weight'] = self.weight
        
        if bias:
            self.bias = Tensor(np.zeros(out_channels), requires_grad=True)
            self._parameters['bias'] = self.bias
        else:
            self.register_parameter('bias', None)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the convolutional layer.
        
        Args:
            x: Input tensor of shape (batch_size, in_channels, height, width)
            
        Returns:
            Output tensor of shape (batch_size, out_channels, out_height, out_width)
        """
        # TODO: Implement convolution operation
        raise NotImplementedError("Conv2d forward pass not implemented yet")


class BatchNorm2d(Module):
    """Batch Normalization layer for 2D inputs.
    
    Args:
        num_features: Number of features/channels
        eps: Small constant for numerical stability. Default: 1e-5
        momentum: Value used for running_mean and running_var computation. Default: 0.1
        affine: If True, has learnable affine parameters. Default: True
        track_running_stats: If True, tracks running mean and variance. Default: True
    """
    
    def __init__(
        self,
        num_features: int,
        eps: float = 1e-5,
        momentum: float = 0.1,
        affine: bool = True,
        track_running_stats: bool = True
    ) -> None:
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        
        if self.affine:
            self.weight = Tensor(np.ones(num_features), requires_grad=True)
            self.bias = Tensor(np.zeros(num_features), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self.bias
            
        if self.track_running_stats:
            self.register_buffer('running_mean', np.zeros(num_features))
            self.register_buffer('running_var', np.ones(num_features))
            self.register_buffer('num_batches_tracked', 0)
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the batch normalization layer.
        
        Args:
            x: Input tensor of shape (batch_size, num_features, height, width)
            
        Returns:
            Normalized tensor of the same shape
        """
        # TODO: Implement batch normalization
        raise NotImplementedError("BatchNorm2d forward pass not implemented yet")
    
class LayerNorm(Module):
    """Layer Normalization layer.
    
    Args:
        normalized_shape: Shape of the input tensor over which to normalize
        eps: Small constant for numerical stability. Default: 1e-5
        elementwise_affine: If True, has learnable affine parameters. Default: True
    """
    
    def __init__(
        self,
        normalized_shape: Tuple[int],
        eps: float = 1e-5,
        elementwise_affine: bool = True
    ) -> None:
        super().__init__()
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        if self.elementwise_affine:
            self.weight = Tensor(np.ones(normalized_shape), requires_grad=True)
            self.bias = Tensor(np.zeros(normalized_shape), requires_grad=True)
            self._parameters['weight'] = self.weight
            self._parameters['bias'] = self.bias
            
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the layer normalization layer.
        
        Args:
            x: Input tensor of shape (*, normalized_shape)
            
        Returns:
            Normalized tensor of the same shape
        """

        # TODO: Implement layer normalization
        pass


# Activation layers that use the implementations from activations.py
class ReLU(Module):
    """ReLU activation layer."""
    
    def __init__(self, inplace: bool = False) -> None:
        super().__init__()
        self.activation = ReLUActivation()
        self.inplace = inplace  # For PyTorch compatibility, not used currently
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)
    
    def extra_repr(self) -> str:
        return f'inplace={self.inplace}'


class LeakyReLU(Module):
    """Leaky ReLU activation layer."""
    
    def __init__(self, negative_slope: float = 0.01, inplace: bool = False) -> None:
        super().__init__()
        self.activation = LeakyReLUActivation(negative_slope)
        self.negative_slope = negative_slope
        self.inplace = inplace  # For PyTorch compatibility, not used currently
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)
    
    def extra_repr(self) -> str:
        return f'negative_slope={self.negative_slope}, inplace={self.inplace}'
    
class ELU(Module):
    """ELU activation layer."""
    
    def __init__(self, alpha: float = 1.0) -> None:
        super().__init__()
        self.activation = ELUActivation(alpha)
        self.alpha = alpha
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)
    
    def extra_repr(self) -> str:
        return f'alpha={self.alpha}'
    
class SELU(Module):
    """SELU activation layer."""
    
    def __init__(self) -> None:
        super().__init__()
        self.activation = SELUActivation()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)

class Sigmoid(Module):
    """Sigmoid activation layer."""
    
    def __init__(self) -> None:
        super().__init__()
        self.activation = SigmoidActivation()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)


class Softmax(Module):
    """Softmax activation layer."""
    
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.activation = SoftmaxActivation(dim)
        self.dim = dim
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)
    
    def extra_repr(self) -> str:
        return f'dim={self.dim}'
    
class Tanh(Module):
    """Tanh activation layer."""
    
    def __init__(self) -> None:
        super().__init__()
        self.activation = TanhActivation()
        
    def forward(self, x: Tensor) -> Tensor:
        return self.activation(x)