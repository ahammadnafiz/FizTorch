import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from fiztorch.tensor import Tensor

t1 = Tensor([1, 2, 3], requires_grad=True)
t2 = Tensor([4, 5, 6], requires_grad=True)

t3 = t1 + t2
t4 = Tensor([2, 0, 1], requires_grad=True)
t5 = t3*t4
t5.backward()
print(t5.grad)