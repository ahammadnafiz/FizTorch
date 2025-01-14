import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from fiztorch.tensor import Tensor

# # Create initial tensors
# x = Tensor([0.0, np.pi / 6, np.pi / 4, np.pi / 2], requires_grad=True)
# y = Tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
# z = Tensor([5.0, 6.0, 7.0, 8.0], requires_grad=True)
#
# # Build a long computational graph
# a = x.sin()                # Sine of x
# b = y * z                  # Element-wise multiplication of y and z
# c = a + b                  # Add a and b
# d = c.exp()                # Exponential of c
# e = d.log()                # Logarithm of d
# f = e.sum() + (x * y).sum()  # Summation of e and x * y
# g = f + (z**2).sum()         # Final operation: add square of z to f
#
# # Print the final result
# print("Final Result (g):", g)
#
# # Perform backward pass to compute gradients
# g.backward()
#
#
# # Check intermediate computations
# print("a (sin(x)):", a)
# print("b (y * z):", b)
# print("c (a + b):", c)
# print("d (exp(c)):", d)
# print("e (log(d)):", e)
# print("f (sum(e) + sum(x * y)):", f)
# print("g (f + sum(z**2)):", g)
#
# # Check gradients after backward
# print("Gradient w.r.t x:", x.grad)
# print("Gradient w.r.t y:", y.grad)
# print("Gradient w.r.t z:", z.grad)


# Create a tensor
x = Tensor([0.0, np.pi / 4, np.pi / 2], requires_grad=True)

# Compute sine
y = x.sin()

# Build a computational graph
z = y + x**2  # Add sine result with squared values

# Print the resulting tensor
print("Result (z):", z)

# Compute gradients
z.backward()

# Display gradients
print("Gradient w.r.t x:", x.grad)
