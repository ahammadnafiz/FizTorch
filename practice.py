import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from fiztorch.tensor import Tensor

a = Tensor([[1, 2, 3]])
b = Tensor([np.arange(3)])

c = a.dot(b)
print(c)
