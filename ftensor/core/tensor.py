import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True):
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
        return Tensor(self.data.sum(axis=axis), requires_grad=self.requires_grad)

    def matmul(self, y):
        return Tensor(np.matmul(self.data, y.data), 
                     requires_grad=self.requires_grad or y.requires_grad)

    def relu(self):
        return Tensor(np.maximum(0, self.data), requires_grad=self.requires_grad)

    def exp(self):
        return Tensor(np.exp(self.data), requires_grad=self.requires_grad)

    def add(self, other):
        return Tensor(self.data + other.data, 
                     requires_grad=self.requires_grad or other.requires_grad)

    def mul(self, other):
        return Tensor(self.data * other.data, 
                     requires_grad=self.requires_grad or other.requires_grad)

    def div(self, other):
        if np.any(other.data == 0):
            raise ZeroDivisionError("Division by zero")
        return Tensor(self.data / other.data, 
                     requires_grad=self.requires_grad or other.requires_grad)

    def sub(self, other):
        return Tensor(self.data - other.data, 
                     requires_grad=self.requires_grad or other.requires_grad)

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

    def backward(self):
        if not self.requires_grad:
            return
            
        if self.grad is None:
            self.grad = Tensor(np.ones_like(self.data))
            
        for node in self.get_topo_graph():
            if node.ctx and node.requires_grad:
                node.ctx.backward(node.grad)

    def get_topo_graph(self):
        topo = []
        visited = set()

        def build_graph(tensor):
            if tensor not in visited and tensor.requires_grad:
                visited.add(tensor)
                if hasattr(tensor, 'ctx'):
                    for parent in tensor.ctx.parents:
                        build_graph(parent)
                topo.append(tensor)

        build_graph(self)
        return reversed(topo)

    def __repr__(self):
        return f"Tensor({self.data}, requires_grad={self.requires_grad})"

    def __getitem__(self, slc):
        return Tensor(self.data[slc], requires_grad=self.requires_grad)

    def __setitem__(self, slc, x):
        self.data[slc] = x.data if isinstance(x, Tensor) else x

    def move(self, x):
        self.data = x.data
        return self

    def pow(self, exponent):
        return Tensor(self.data ** exponent, requires_grad=self.requires_grad)

    def zero_grad(self):
        """Zero out the gradient"""
        self.grad = None

    def detach(self):
        """Returns a new Tensor, detached from the current graph"""
        return Tensor(self.data, requires_grad=False)

# Automatically define operators for Tensor
for op in ['add', 'mul', 'sub', 'pow']:
    setattr(Tensor, f"__{op}__", Tensor.totensor(getattr(Tensor, op)))
    setattr(Tensor, f"__r{op}__", Tensor.totensor(getattr(Tensor, op)))
    setattr(Tensor, f"__i{op}__", lambda self, x, op=op: self.move(getattr(Tensor, op)(self, x)))