import sys
import os

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from ftensor.core.ftensor import FTensor as ft



# Creating higher-dimensional tensors
a = ft([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
b = ft([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])  # Shape: (2, 2, 2)

# Basic operations
print("a + b:\n", a + b)                          # Element-wise addition
print("a - b:\n", a - b)                          # Element-wise subtraction
print("a * b:\n", a * b)                          # Element-wise multiplication
print()

# Dot product
result = a.dot(b)
print("a.dot(b):\n", result)                      # Dot product
print()

# Flattening and transposing
print("a.flatten():\n", a.flatten())              # Flatten the tensor
print("b.transpose():\n", b.transpose())          # Transpose the tensor
print()

# One-dimensional tensor testing
c = ft([[1, 2, 4]])                               # Shape: (1, 3)
print("c.shape:", c.shape)                        # Shape of c
print("c.transpose().shape:", c.transpose().shape)  # Shape after transpose
print()

# Two-dimensional tensor testing
d = ft([[1, 2, 3], [4, 5, 6]])                   # Shape: (2, 3)
print("d.sum(axis=0):", d.sum(axis=0))            # Sum over rows
print("d.sum(axis=1):", d.sum(axis=1))            # Sum over columns
print("d.sum():", d.sum())                        # Total sum
print("b.size:", b.size)                          # Size of tensor b
print()

# Testing with different types of tensors
c = ft([1.0, 2.0, -1.0, 8.0])                    # Shape: (4,)
d = ft([1.0, 2.0, -1.0, 0.0])                    # Shape: (4,)

print("c + d:\n", c + d)                          # Element-wise addition
print("c.log():\n", c.log())                      # Logarithm
print("a.log():\n", a.log())                      # Logarithm of tensor a
print("c.exp():\n", c.exp())                      # Exponential
print("d.exp():\n", d.exp())                      # Exponential
print("d.softmax():\n", d.softmax())              # Softmax
print("a.relu_derivative():\n", a.relu_derivative())  # ReLU derivative
print()

# More complex operations with reshaping
e = ft([[1, 2], [3, 4], [5, 6]])                  # Shape: (3, 2)
f = e.reshape((2, 3))                             # Reshape to (2, 3)
print("e reshaped to (2, 3):\n", f)
print()

# Further tensor manipulations
g = ft([[[1], [2]], [[3], [4]], [[5], [6]]])    # Shape: (3, 2, 1)
h = g.sum(axis=0)                                 # Sum over the first axis
print("g.sum(axis=0):\n", h)
print()

# Testing broadcasting with tensors of different shapes
i = ft([[1, 2], [3, 4]])                          # Shape: (2, 2)
j = ft([1, 2])                                    # Shape: (2,)
print()

print("i + j:\n", i + j)                          # Broadcasting addition
print("\nj + i:\n", j + i)

print()
def run_tests(tests):
    for i, (desc, a, b) in enumerate(tests, 1):
        print(f"\nTest {i}: {desc}")
        ft_a = ft(a)
        ft_b = ft(b)
        np_a = np.array(a)
        np_b = np.array(b)
        
        ft_result = ft_a + ft_b
        np_result = np_a + np_b
        
        print(f"FTensor result:\n{ft_result}")
        print(f"NumPy result:\n{np_result}")
        
        if np.array_equal(ft_result.data, np_result):
            print("Test passed: Results match!")
        else:
            print("Test failed: Results do not match.")

# Define test cases: (description, tensor_a, tensor_b)
tests = [
    ("2D + 1D", [[1, 2], [3, 4]], [10, 20]),
    ("3D + 2D", [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[10, 20], [30, 40]]),
    ("2D + scalar", [[1, 2], [3, 4]], 10),
    ("3D + 1D", [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [10, 20]),
    ("4D + 3D", [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], [[[10, 20]], [[30, 40]]]),
    ("2D + 2D (different shapes)", [[1, 2, 3], [4, 5, 6]], [[1], [2]]),
    ("3D + 2D", [[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[2, 2], [2, 2]]),
    ("1D + 3D", [1, 2], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ("2D + 3D", [[1, 2], [3, 4]], [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
    ("4D + scalar", [[[[1, 2], [3, 4]]], [[[5, 6], [7, 8]]]], 10),
]

# Run the tests
run_tests(tests)
print()