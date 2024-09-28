from __future__ import annotations

from ..core.function import Function
from ..core.tensor import Tensor, FTensor
import numpy as np

class Add(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(np.add(a.data, b.data))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, grad_output

class Mul(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(np.multiply(a.data, b.data))

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output * b.data, grad_output * a.data

class Sub(Function):
    @staticmethod
    def forward(ctx, a, b):
        return Tensor(np.subtract(a.data, b.data))

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, -grad_output

class Div(Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.save_for_backward(a, b)
        return Tensor(np.divide(a.data, b.data))

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return grad_output / b.data, -grad_output * a.data / (b.data ** 2)

class Sum(Function):
    @staticmethod
    def forward(ctx, input, axis=None):
        ctx.axis = axis
        ctx.input_shape = input.shape
        return Tensor(np.sum(input.data, axis=axis))

    @staticmethod
    def backward(ctx, grad_output):
        return np.broadcast_to(grad_output, ctx.input_shape)

class Mean(Function):
    @staticmethod
    def forward(ctx, input, axis=None):
        ctx.axis = axis
        ctx.input_shape = input.shape
        return Tensor(np.mean(input.data, axis=axis))

    @staticmethod
    def backward(ctx, grad_output):
        output_shape = grad_output.shape
        scale = np.prod(ctx.input_shape) / np.prod(output_shape)
        return np.broadcast_to(grad_output, ctx.input_shape) / scale

class Dot(Function):
    @staticmethod
    def forward(ctx, a, b):
        if a.shape[1] != b.shape[0]:
            raise ValueError(f"Incompatible shapes for dot product: {a.shape} and {b.shape}")
        return np.dot(a.data, b.data)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        return np.dot(grad_output, b.data.T), np.dot(a.data.T, grad_output)

class Transpose(Function):
    @staticmethod
    def forward(ctx, input):
        return Tensor(np.transpose(input.data))

    @staticmethod
    def backward(ctx, grad_output):
        return np.transpose(grad_output)

class Reshape(Function):
    @staticmethod
    def forward(ctx, input, new_shape):
        ctx.input_shape = input.shape
        reshaped_data = np.reshape(input.data, new_shape)
        return FTensor(reshaped_data)

    @staticmethod
    def backward(ctx, grad_output):
        return np.reshape(grad_output, ctx.input_shape)

class Log(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return Tensor(np.log(input.data))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output / input.data

class Exp(Function):
    @staticmethod
    def forward(ctx, input):
        output = Tensor(np.exp(input.data))
        ctx.save_for_backward(output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * output.data

class ReLU(Function):
    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return Tensor(np.maximum(input.data, 0))

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return grad_output * (input.data > 0)

class Sigmoid(Function):
    @staticmethod
    def forward(ctx, input):
        output = 1 / (1 + np.exp(-input.data))
        ctx.save_for_backward(Tensor(output))
        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * output.data * (1 - output.data)

class Tanh(Function):
    @staticmethod
    def forward(ctx, input):
        output = np.tanh(input.data)
        ctx.save_for_backward(Tensor(output))
        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        return grad_output * (1 - output.data ** 2)

class Softmax(Function):
    @staticmethod
    def forward(ctx, input, axis=-1):
        exp = np.exp(input.data - np.max(input.data, axis=axis, keepdims=True))
        output = exp / np.sum(exp, axis=axis, keepdims=True)
        ctx.save_for_backward(Tensor(output), axis)
        return Tensor(output)

    @staticmethod
    def backward(ctx, grad_output):
        output, axis = ctx.saved_tensors
        return grad_output * output.data - output.data * np.sum(grad_output * output.data, axis=axis, keepdims=True)