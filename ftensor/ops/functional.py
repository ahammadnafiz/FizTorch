from __future__ import annotations
from typing import Optional, Tuple, Union, Any
import numpy as np

from ..core.function import Function
from ..core.tensor import Tensor
from .tensorops import TensorOps

class MatMul(Function):
    """Matrix multiplication operation."""
    
    def forward(ctx, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computes x1 @ x2."""
        return Tensor(np.matmul(x1.data, x2.data), 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass computes gradients for matrix multiplication."""
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += grad_out.matmul(x2.transpose())
        if x2.requires_grad:
            x2.grad += x1.transpose().matmul(grad_out)

class Sum(Function):
    """Summation operation along specified axes."""
    
    def forward(ctx, x1: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass computes sum along specified axes."""
        if "axis" in kwargs:
            _axis = kwargs['axis'] if isinstance(kwargs['axis'], tuple) else (kwargs['axis'],)
            # Handle negative axes
            ctx.axis = tuple(d if d >= 0 else (len(x1.shape) + d) for d in _axis)
        else:
            ctx.axis = None
        kwargs["keepdims"] = True
        return Tensor(np.sum(x1.data, **kwargs), 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass extends gradients to match input shape."""
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += TensorOps.extend(grad_out, ctx.parents[0].shape, ctx.axis)

class Add(Function):
    """Element-wise addition operation."""
    
    def forward(ctx, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computes x1 + x2."""
        return Tensor(x1.data + x2.data, 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass broadcasts gradients to both inputs."""
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += TensorOps.broadcast(grad_out, x1.shape)
        if x2.requires_grad:
            x2.grad += TensorOps.broadcast(grad_out, x2.shape)

class Max(Function):
    """Maximum value operation along specified axis."""
    
    def forward(ctx, x1: Tensor, **kwargs: Any) -> Tensor:
        """Forward pass computes maximum values and stores indices."""
        tmp = np.max(x1.data, **kwargs)
        ctx.max_idx = np.argmax(x1.data, **kwargs)
        ctx.axis = kwargs.get("axis", None)
        return Tensor(tmp, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass routes gradients through maximum indices."""
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += TensorOps.max_broad(ctx.max_idx, grad_out, ctx.axis, ctx.parents[0].shape)

class Exp(Function):
    """Exponential operation."""
    
    def forward(ctx, x1: Tensor) -> Tensor:
        """Forward pass computes exp(x1) and stores result."""
        ret = Tensor(np.exp(x1.data), requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.save_for_backward(ret)
        return ret

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass multiplies gradient with exp(x1)."""
        if ctx.parents[0].requires_grad:
            saved_output = ctx.get_saved_tensors()[0]
            ctx.parents[0].grad += grad_out * saved_output

class Sub(Function):
    """Element-wise subtraction operation."""
    
    def forward(ctx, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computes x1 - x2."""
        return Tensor(x1.data - x2.data, 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass handles gradients for subtraction."""
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += TensorOps.broadcast(grad_out, x1.shape)
        if x2.requires_grad:
            x2.grad += TensorOps.broadcast(-1 * grad_out, x2.shape)

class Mul(Function):
    """Element-wise multiplication operation."""
    
    def forward(ctx, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computes x1 * x2."""
        return Tensor(x1.data * x2.data, 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass applies product rule."""
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += TensorOps.broadcast(grad_out * x2, x1.shape)
        if x2.requires_grad:
            x2.grad += TensorOps.broadcast(grad_out * x1, x2.shape)

class ReLU(Function):
    """Rectified Linear Unit activation function."""
    
    def forward(ctx, x1: Tensor) -> Tensor:
        """Forward pass computes max(0, x1) and stores mask."""
        mask = x1.data > 0
        ctx.save_for_backward(mask)
        return Tensor(x1.data * mask, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass zeros out gradient where input was negative."""
        if ctx.parents[0].requires_grad:
            mask = ctx.get_saved_tensors()[0]
            ctx.parents[0].grad[mask] += grad_out[mask]

class Pow(Function):
    """Power operation x1 ** x2."""
    
    def forward(ctx, x1: Tensor, x2: Tensor) -> Tensor:
        """Forward pass computes x1 ** x2 and stores result."""
        ret = Tensor(np.power(x1.data, x2.data), 
                    requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.save_for_backward(ret)
        return ret

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass applies chain rule for power operation."""
        x1, x2 = ctx.parents
        saved_output = ctx.get_saved_tensors()[0]
        if x1.requires_grad:
            x1.grad += TensorOps.broadcast(
                grad_out * (x2 * x1 ** (x2 - 1)), x1.shape)
        if x2.requires_grad:
            x2.grad += TensorOps.broadcast(
                grad_out * x1.log() * saved_output, x2.shape)

class Log(Function):
    """Natural logarithm operation."""
    
    def forward(ctx, x: Tensor) -> Tensor:
        """Forward pass computes log(x) with numerical stability."""
        # Clamp minimum values for numerical stability
        x.data[x.data <= 1e-7] = 1e-7
        return Tensor(np.log(x.data), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass computes gradient as 1/x."""
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += grad_out * (ctx.parents[0] ** -1)

class Slice(Function):
    """Slicing operation."""
    
    def forward(ctx, x: Tensor, *args: Any) -> Tensor:
        """Forward pass applies slice operation."""
        ctx.save_for_backward(args)
        return Tensor(x.data.__getitem__(*args), 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass accumulates gradients at sliced positions."""
        args = ctx.get_saved_tensors()[0]
        ctx.parents[0].grad[args] += grad_out

class Permute(Function):
    """Permute dimensions of a tensor."""
    
    def forward(ctx, x: Tensor, order: Tuple[int, ...]) -> Tensor:
        """Forward pass permutes dimensions and stores inverse order."""
        ctx.save_for_backward(np.argsort(order))
        return Tensor(np.moveaxis(x.data, order, tuple(range(x.data.ndim))), 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass applies inverse permutation."""
        order = ctx.get_saved_tensors()[0]
        ctx.parents[0].grad += grad_out.permute(order)

class Reshape(Function):
    """Reshape tensor to new shape."""
    
    def forward(ctx, x: Tensor, shape: Tuple[int, ...]) -> Tensor:
        """Forward pass reshapes tensor and stores original shape."""
        ctx.save_for_backward((shape, x.shape))
        return Tensor(np.reshape(x.data, shape), 
                     requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass reshapes gradient to original shape."""
        _, original_shape = ctx.get_saved_tensors()[0]
        ctx.parents[0].grad += grad_out.reshape(original_shape)

class Embed(Function):
    """Embedding layer operation."""
    
    def forward(ctx, x: Tensor, idx: Tensor, *, n_embd: int) -> Tensor:
        """Forward pass performs embedding lookup."""
        ctx.save_for_backward(idx)
        out = Tensor.zeros((idx.shape[0], idx.shape[1], n_embd), requires_grad=True)
        for bdim in range(idx.shape[0]):
            out[bdim] = x[idx[bdim].data]
        return Tensor(out.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out: Tensor) -> None:
        """Backward pass accumulates gradients for each embedding vector."""
        idx = ctx.get_saved_tensors()[0]
        grad = ctx.parents[0].grad
        for bdim in range(idx.shape[0]):
            for row, dx in enumerate(idx[bdim]):
                grad[dx.data] += grad_out[bdim, row]