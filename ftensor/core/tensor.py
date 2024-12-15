import numpy as np

class Context:
    def __init__(self):
        self.saved_tensors = []
        self.parents = []

    def save_for_backward(self, *tensors):
        self.saved_tensors.extend(tensors)

class AddContext(Context):
    def __init__(self, a, b):
        super().__init__()
        self.parents = [a, b]
    
    def backward(self, grad_output):
        a, b = self.parents
        if a.requires_grad:
            if a.grad is None:
                a.grad = Tensor(np.zeros_like(a.data))
            a.grad.data += grad_output.data
        if b.requires_grad:
            if b.grad is None:
                b.grad = Tensor(np.zeros_like(b.data))
            b.grad.data += grad_output.data

class MulContext(Context):
    def __init__(self, a, b):
        super().__init__()
        self.parents = [a, b]
        self.save_for_backward(a, b)
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if a.requires_grad:
            if a.grad is None:
                a.grad = Tensor(np.zeros_like(a.data))
            a.grad.data += grad_output.data * b.data
        if b.requires_grad:
            if b.grad is None:
                b.grad = Tensor(np.zeros_like(b.data))
            b.grad.data += grad_output.data * a.data

class MatMulContext(Context):
    def __init__(self, a, b):
        super().__init__()
        self.parents = [a, b]
        self.save_for_backward(a, b)
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if a.requires_grad:
            if a.grad is None:
                a.grad = Tensor(np.zeros_like(a.data))
            a.grad.data += np.matmul(grad_output.data, b.data.T)
        if b.requires_grad:
            if b.grad is None:
                b.grad = Tensor(np.zeros_like(b.data))
            b.grad.data += np.matmul(a.data.T, grad_output.data)

class SubContext(Context):
    def __init__(self, a, b):
        super().__init__()
        self.parents = [a, b]
    
    def backward(self, grad_output):
        a, b = self.parents
        if a.requires_grad:
            if a.grad is None:
                a.grad = Tensor(np.zeros_like(a.data))
            a.grad.data += grad_output.data
        if b.requires_grad:
            if b.grad is None:
                b.grad = Tensor(np.zeros_like(b.data))
            b.grad.data -= grad_output.data

class DivContext(Context):
    def __init__(self, a, b):
        super().__init__()
        self.parents = [a, b]
        self.save_for_backward(a, b)
    
    def backward(self, grad_output):
        a, b = self.saved_tensors
        if a.requires_grad:
            if a.grad is None:
                a.grad = Tensor(np.zeros_like(a.data))
            a.grad.data += grad_output.data / b.data
        if b.requires_grad:
            if b.grad is None:
                b.grad = Tensor(np.zeros_like(b.data))
            b.grad.data += -grad_output.data * a.data / (b.data * b.data)

class PowContext(Context):
    def __init__(self, a, exponent):
        super().__init__()
        self.parents = [a]
        self.exponent = exponent
        self.save_for_backward(a)
    
    def backward(self, grad_output):
        a, = self.saved_tensors
        if a.requires_grad:
            if a.grad is None:
                a.grad = Tensor(np.zeros_like(a.data))
            a.grad.data += grad_output.data * self.exponent * np.power(a.data, self.exponent - 1)

class ReLUContext(Context):
    def __init__(self, x):
        super().__init__()
        self.parents = [x]
        self.save_for_backward(x)
    
    def backward(self, grad_output):
        x, = self.saved_tensors
        if x.requires_grad:
            if x.grad is None:
                x.grad = Tensor(np.zeros_like(x.data))
            x.grad.data += grad_output.data * (x.data > 0)

class ExpContext(Context):
    def __init__(self, x):
        super().__init__()
        self.parents = [x]
        self.save_for_backward(x)
        self.result = None
    
    def backward(self, grad_output):
        x, = self.saved_tensors
        if x.requires_grad:
            if x.grad is None:
                x.grad = Tensor(np.zeros_like(x.data))
            x.grad.data += grad_output.data * np.exp(x.data)
            
# Add SumContext for handling sum operation gradients
class SumContext(Context):
    def __init__(self, x, axis=None):
        super().__init__()
        self.parents = [x]
        self.axis = axis
        self.shape = x.shape
    
    def backward(self, grad_output):
        x, = self.parents
        if x.requires_grad:
            if x.grad is None:
                x.grad = Tensor(np.zeros_like(x.data))
            # Reshape grad_output to match original tensor shape
            if self.axis is not None:
                grad_shape = list(self.shape)
                grad_shape[self.axis] = 1
                grad_broadcast = np.reshape(grad_output.data, grad_shape)
                x.grad.data += np.broadcast_to(grad_broadcast, self.shape)
            else:
                x.grad.data += np.broadcast_to(grad_output.data, self.shape)

class Tensor:
    def __init__(self, data, requires_grad=False):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None if requires_grad else None
        self.ctx = None

    @property
    def shape(self):
        return self.data.shape

    @classmethod
    def eye(cls, val):
        return cls(np.eye(val))

    @classmethod
    def ones(cls, shape, requires_grad=False):
        return cls(np.ones(shape), requires_grad=requires_grad)

    @classmethod
    def zeros(cls, shape, requires_grad=False):
        return cls(np.zeros(shape), requires_grad=requires_grad)

    @staticmethod
    def totensor(func):
        def inner(*args):
            args = [Tensor(arg, requires_grad=False) if isinstance(arg, (int, float)) else arg for arg in args]
            return func(*args)
        return inner

    def flatten(self):
        return Tensor(self.data.flatten(), requires_grad=self.requires_grad)

    def cpu(self):
        return self.data

    def sum(self, axis=None):
        ret = Tensor(self.data.sum(axis=axis), requires_grad=self.requires_grad)
        if ret.requires_grad:
            ret.ctx = SumContext(self, axis)
        return ret

    def matmul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        ret = Tensor(np.matmul(self.data, other.data), 
                    requires_grad=self.requires_grad or other.requires_grad)
        if ret.requires_grad:
            ret.ctx = MatMulContext(self, other)
        return ret

    def relu(self):
        ret = Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)
        if ret.requires_grad:
            ret.ctx = ReLUContext(self)
        return ret

    def exp(self):
        ret = Tensor(np.exp(self.data), requires_grad=self.requires_grad)
        if ret.requires_grad:
            ret.ctx = ExpContext(self)
        return ret

    def add(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        ret = Tensor(self.data + other.data, 
                    requires_grad=self.requires_grad or other.requires_grad)
        if ret.requires_grad:
            ret.ctx = AddContext(self, other)
        return ret

    def mul(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        ret = Tensor(self.data * other.data, 
                    requires_grad=self.requires_grad or other.requires_grad)
        if ret.requires_grad:
            ret.ctx = MulContext(self, other)
        return ret

    def div(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        if np.any(other.data == 0):
            raise ZeroDivisionError("Division by zero")
        ret = Tensor(self.data / other.data, 
                    requires_grad=self.requires_grad or other.requires_grad)
        if ret.requires_grad:
            ret.ctx = DivContext(self, other)
        return ret

    def sub(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        ret = Tensor(self.data - other.data, 
                    requires_grad=self.requires_grad or other.requires_grad)
        if ret.requires_grad:
            ret.ctx = SubContext(self, other)
        return ret

    def permute(self, orders):
        return Tensor(np.transpose(self.data, orders), requires_grad=self.requires_grad)

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape), requires_grad=self.requires_grad)

    def transpose(self, dim0=-2, dim1=-1):
        axes = list(range(len(self.shape)))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self.data, axes), requires_grad=self.requires_grad)

    def dropout(self, p):
        if not 0 <= p < 1:
            raise ValueError("Dropout probability must be between 0 and 1")
        mask = np.random.binomial(1, 1 - p, self.shape)
        result = Tensor(self.data * mask * (1 / (1 - p)), requires_grad=self.requires_grad)
        result.mask = mask  # Save mask for backward pass
        return result

    def backward(self, grad_output=None):
        if not self.requires_grad:
            return

        if grad_output is None:
            grad_output = Tensor(np.ones_like(self.data))
            
        if self.grad is None:
            self.grad = Tensor(np.zeros_like(self.data))
        self.grad.data += grad_output.data

        topo = []
        visited = set()

        def build_topo(tensor):
            if tensor not in visited and tensor.requires_grad:
                visited.add(tensor)
                if tensor.ctx:
                    for parent in tensor.ctx.parents:
                        build_topo(parent)
                topo.append(tensor)

        build_topo(self)

        for node in reversed(topo):
            if node.ctx:
                node.ctx.backward(node.grad)

    def pow(self, exponent):
        ret = Tensor(self.data ** exponent, requires_grad=self.requires_grad)
        if ret.requires_grad:
            ret.ctx = PowContext(self, exponent)
        return ret

    def zero_grad(self):
        """Zero out the gradient"""
        self.grad = None

    def detach(self):
        """Returns a new Tensor, detached from the current graph"""
        return Tensor(self.data, requires_grad=False)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __getitem__(self, slc):
        return Tensor(self.data[slc], requires_grad=self.requires_grad)

    def __setitem__(self, slc, x):
        self.data[slc] = x.data if isinstance(x, Tensor) else x

    def move(self, x):
        self.data = x.data
        return self

    def __add__(self, other): return self.add(other)
    def __mul__(self, other): return self.mul(other)
    def __sub__(self, other): return self.sub(other)
    def __truediv__(self, other): return self.div(other)
    def __pow__(self, other): return self.pow(other)
    def __matmul__(self, other): return self.matmul(other)
    
    def __radd__(self, other): return self.add(other)
    def __rmul__(self, other): return self.mul(other)
    def __rsub__(self, other): return Tensor(other).sub(self)
    def __rtruediv__(self, other): return Tensor(other).div(self)
    def __rpow__(self, other): return Tensor(other).pow(self)
