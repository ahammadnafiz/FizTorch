import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fiztorch.tensor import Tensor

a = Tensor([1, 2, 3, 4, 5])
b = Tensor([1, 2, 3, 4, 5])

a = a.to_float32()
b = b.to_float64()
print(a)
print(b)