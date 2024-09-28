from ..core.tensor import Tensor
import numpy as np

class TensorOps:
    @staticmethod
    def max_broad(max_idx, grad_out, axis, out_shape):
        shape = max_idx.shape
        grad = np.zeros(out_shape)
        dims_strides = np.cumprod((1, *shape[::-1]))[::-1]
        dims = np.zeros_like(shape)
        if axis is None:
            grad0 = grad.reshape(-1)
            grad0[max_idx] = grad_out.data
        else:
            for v, (i, d) in enumerate(zip(max_idx.flatten(), grad_out.flatten())):
                for j in range(len(shape)):
                    dims[j] = (v // dims_strides[j + 1]) % shape[j]
                dims[axis] = i
                grad[tuple(dims)] = d
        return Tensor(grad, requires_grad=False)

    @staticmethod
    def extend(data, shape, axis):
        ext_shape = tuple(s if idx in axis else 1 for idx, s in enumerate(shape)) if axis else shape
        return Tensor(np.tile(data.data, ext_shape), requires_grad=data.requires_grad)

    @staticmethod
    def broadcast(data, shape):
        src_shape = data.shape
        if src_shape == shape:
            return data
        if len(shape) != len(src_shape):
            brd, kd = tuple(range(len(src_shape) - len(shape))), False
        else:
            brd, kd = tuple(idx for idx, (s, d) in enumerate(zip(src_shape, shape)) if s != d), True
        return Tensor(data.data.sum(axis=brd, keepdims=kd), requires_grad=data.requires_grad)