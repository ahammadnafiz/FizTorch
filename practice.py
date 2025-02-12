import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fiztorch.tensor import Tensor

# 1. Create a tensor
a = Tensor(np.random.randn(2, 3), requires_grad=True)
b = Tensor(np.random.randn(2, 3), requires_grad=True)
w = Tensor(np.random.randn(3, 4), requires_grad=True)

# 2. Perform operations
c = a.dot(b) + w
d = c.relu()
d.backward()

print(a.grad)
print(b.grad)
# print(w.grad)
# print(c)
# print(d)
