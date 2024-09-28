from __future__ import annotations

from ..core.function import Function
from ..core.tensor import Tensor
from tensorops import max_broad, extend, broadcast
import numpy as np

class MATMUL(Function):
    def forward(ctx, x1, x2):
        return Tensor(np.matmul(x1.data, x2.data), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += grad_out.matmul(x2.transpose())
        if x2.requires_grad:
            x2.grad += x1.transpose().matmul(grad_out)

class SUM(Function):
    def forward(ctx, x1, **kwargs):
        if "axis" in kwargs:
            _axis = kwargs['axis'] if isinstance(kwargs['axis'], tuple) else (kwargs['axis'],)
            ctx.axis = tuple(d if d >= 0 else (len(x1.shape) - d) for d in _axis)
        else:
            ctx.axis = None
        kwargs["keepdims"] = True
        return Tensor(np.sum(x1.data, **kwargs), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += extend(grad_out, ctx.parents[0].shape, ctx.axis)

class ADD(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data + x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += broadcast(grad_out, x1.shape)
        if x2.requires_grad:
            x2.grad += broadcast(grad_out, x2.shape)

class MAX(Function):
    def forward(ctx, x1, **kwargs):
        tmp = np.max(x1.data, **kwargs)
        ctx.max_idx = np.argmax(x1.data, **kwargs)
        ctx.axis = kwargs.get("axis", None)
        return Tensor(tmp, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += max_broad(ctx.max_idx, grad_out, ctx.axis, ctx.parents[0].shape)

class EXP(Function):
    def forward(ctx, x1):
        ret = Tensor(np.exp(x1.data), requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.outs.append(ret)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += grad_out * ctx.outs[0]

class SUB(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data - x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += broadcast(grad_out, x1.shape)
        if x2.requires_grad:
            x2.grad += broadcast(-1 * grad_out, x2.shape)

class MUL(Function):
    def forward(ctx, x1, x2):
        return Tensor(x1.data * x2.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        x1, x2 = ctx.parents
        if x1.requires_grad:
            x1.grad += broadcast(grad_out * x2, x1.shape)
        if x2.requires_grad:
            x2.grad += broadcast(grad_out * x1, x2.shape)

class RELU(Function):
    def forward(ctx, x1):
        ctx.outs.append(x1.data > 0)
        return Tensor(x1.data * (x1.data > 0), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        idx, x = ctx.outs[0], ctx.parents[0]
        if x.requires_grad:
            x.grad[idx] += grad_out[idx]

class POW(Function):
    def forward(ctx, x1, x2):
        ret = Tensor(np.power(x1.data, x2.data), requires_grad=ctx.requires_grad, ctx=ctx)
        ctx.outs.append(ret)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += broadcast(grad_out * (ctx.parents[1] * ctx.parents[0] ** (ctx.parents[1] - 1)), ctx.parents[0].shape)
        if ctx.parents[1].requires_grad:
            ctx.parents[1].grad += broadcast(grad_out * ctx.parents[0].log() * ctx.outs[0], ctx.parents[1].shape)

class LOG(Function):
    def forward(ctx, x):
        x.data[x.data <= 1e-7] = 1e-7
        ret = Tensor(np.log(x.data), requires_grad=ctx.requires_grad, ctx=ctx)
        return ret

    def backward(ctx, grad_out):
        if ctx.parents[0].requires_grad:
            ctx.parents[0].grad += grad_out * (ctx.parents[0] ** -1)

class SLC(Function):
    def forward(ctx, x, *args):
        ctx.outs.append(*args)
        return Tensor(x.data.__getitem__(*args), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        args = ctx.outs.pop()
        ctx.parents[0].grad[args] += grad_out

class PERMUTE(Function):
    def forward(ctx, x, order):
        ctx.outs.append(np.argsort(order))
        return Tensor(np.moveaxis(x.data, order, tuple(range(x.data.ndim))), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        order = ctx.outs.pop()
        ctx.parents[0].grad += grad_out.permute(order)

class RESHAPE(Function):
    def forward(ctx, x, shape):
        ctx.outs.append((shape, x.shape))
        return Tensor(np.reshape(x.data, shape), requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        _, oshape = ctx.outs.pop()
        ctx.parents[0].grad += grad_out.reshape(oshape)

class EMBED(Function):
    def forward(ctx, x, idx, *, n_embd):
        ctx.outs.append(idx)
        out = Tensor.zeros((idx.shape[0], idx.shape[1], n_embd), requires_grad=True)
        for bdim in range(idx.shape[0]):
            out[bdim] = x[idx[bdim].data]
        return Tensor(out.data, requires_grad=ctx.requires_grad, ctx=ctx)

    def backward(ctx, grad_out):
        idx = ctx.outs.pop()
        grad = ctx.parents[0].grad
        for bdim in range(idx.shape[0]):
            for row, dx in enumerate(idx[bdim]):
                grad[dx.data] += grad_out[bdim, row]