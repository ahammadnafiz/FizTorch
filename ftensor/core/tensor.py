import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=True):
        self.data = np.array(data, dtype=np.float32)
        self.requires_grad = requires_grad
        self.grad = None

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
            args = [Tensor(arg) if isinstance(arg, (int, float)) else arg for arg in args]
            return func(*args)
        return inner

    def flatten(self):
        return Tensor(self.data.flatten())

    def cpu(self):
        return self.data

    def sum(self, axis=None):
        return Tensor(self.data.sum(axis=axis))

    def matmul(self, y):
        return Tensor(np.matmul(self.data, y.data))

    def relu(self):
        return Tensor(np.maximum(0, self.data))

    def exp(self):
        return Tensor(np.exp(self.data))

    def add(self, other):
        return Tensor(self.data + other.data)

    def mul(self, other):
        return Tensor(self.data * other.data)

    def div(self, other):
        return Tensor(self.data / other.data)

    def sub(self, other):
        return Tensor(self.data - other.data)

    def permute(self, orders):
        return Tensor(np.transpose(self.data, orders))

    def reshape(self, shape):
        return Tensor(self.data.reshape(shape))

    def transpose(self, dim0=-2, dim1=-1):
        axes = list(range(len(self.shape)))
        axes[dim0], axes[dim1] = axes[dim1], axes[dim0]
        return Tensor(np.transpose(self.data, axes))

    def dropout(self, p):
        mask = np.random.binomial(1, 1 - p, self.shape)
        return Tensor(self.data * mask * (1 / (1 - p)))

    def backward(self):
        self.grad = Tensor(np.ones_like(self.data))
        for node in self.get_topo_graph():
            if node.ctx:
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
        return f"Tensor({self.data})"

    def __getitem__(self, slc):
        return Tensor(self.data[slc])

    def __setitem__(self, slc, x):
        self.data[slc] = x.data

    def move(self, x):
        self.data = x.data
        return self

    def pow(self, exponent):
        # Implement the power operation
        return self.data ** exponent

# Automatically define operators for Tensor
for op in ['add', 'mul', 'sub', 'pow']:
    setattr(Tensor, f"__{op}__", Tensor.totensor(getattr(Tensor, op)))
    setattr(Tensor, f"__r{op}__", Tensor.totensor(getattr(Tensor, op)))
    setattr(Tensor, f"__i{op}__", lambda self, x: self.move(getattr(Tensor, op)(self, x)))