![Logo](assets/fiztorch.png)

## Introduction ✨
Popular machine learning libraries like PyTorch and TensorFlow inspire this toy implementation of an FTensor class. This project aims to create a basic tensor class with essential operations and functionality, helping you understand the core concepts behind tensor-based machine learning.

## Key Features 🔑
- ➕ Basic tensor operations: addition, subtraction, multiplication, division
- 🧮 Matrix multiplication
- 📏 Handling of tensor shapes and sizes
- 🔄 Gradient tracking and backpropagation
- 💾 Serialization and persistence
- 🧪 Unit testing and comprehensive documentation

---
## Examples 🚀
``` python
from ftensor import FTensor as ft


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
```

## Implementation Goals 🎯

### Completed Functionalities ✅

- ✅ **Higher-Dimensional Tensors**: Successfully created and manipulated tensors of shape (2, 2, 2).
- ✅ **Basic Operations**:
  - Element-wise addition, subtraction, and multiplication.
- ✅ **Dot Product**: Implemented dot product functionality for tensor operations.
- ✅ **Flattening and Transposing**: Added methods to flatten and transpose tensors.
- ✅ **1D and 2D Tensor Testing**:
  - Successfully tested and displayed shapes of one-dimensional and two-dimensional tensors.
  - Implemented sum operations across specified axes.
- ✅ **Element-wise Operations**: 
  - Implemented element-wise addition, logarithm, exponential, and softmax operations.
- ✅ **ReLU Derivative**: Added functionality for computing ReLU derivatives.
- ✅ **Reshaping Tensors**: Reshape tensors to desired dimensions effectively.
- ✅ **Advanced Tensor Manipulations**:
  - Implemented tensor summation over specified axes.
- ✅ **Testing Framework**: Developed a testing framework to compare results with NumPy.

### Known Issues ⚠️

- ⚠️ **Broadcasting Support**: Some broadcasting operations are not functioning as expected and require further debugging.

```output
Test 5: 4D + 3D
FTensor result:
FTensor(shape=(2, 2, 1, 1, 2), dtype=list)
Data:
[[[[11, 22]]] [[[15, 26]]]]
[[[[11, 22]]] [[[15, 26]]]]
NumPy result:
[[[[11 22]
   [13 24]]

  [[31 42]
   [33 44]]]


 [[[15 26]
   [17 28]]

  [[35 46]
   [37 48]]]]
Test failed: Results do not match.

Test 6: 2D + 2D (different shapes)
FTensor result:
FTensor(shape=(2, 2, 1), dtype=list)
Data:
[[2] [4]]
[[5] [7]]
NumPy result:
[[2 3 4]
 [6 7 8]]
Test failed: Results do not match.
```

### Future Additions and Features 🚀

- 🔲 **Automatic Differentiation**: Implementing backpropagation for automatic gradient calculation.
- 🔲 **Additional Tensor Operations**: Expanding operations to include matrix factorization, eigenvalues, etc.
- 🔲 **Support for Sparse Tensors**: Enhancing functionality to handle sparse tensor representations.
- 🔲 **GPU Acceleration**: Integrating support for GPU computations for performance improvement.
- 🔲 **Comprehensive Documentation**: Providing detailed usage examples and API documentation.
- 🔲 **Unit Testing**: Expanding the testing framework to cover more edge cases and functionality.
- 🔲 **Optimizers**: Implementing various optimization algorithms (e.g., SGD, Adam).
- 🔲 **Performance Benchmarks**: Creating benchmarks to evaluate performance against other frameworks like NumPy.

---

## Getting Started 🚀
To get started, clone the repository and set up your development environment. You'll need Python 3.x installed on your system.

```bash
git clone https://github.com/ahammadnafiz/FizTorch.git
cd FizTorch
```

## Contributing 🤝
Contributions are welcome! If you'd like to help, please follow these guidelines:

1. 🍴 Fork the repository
2. 🌿 Create a new branch for your feature or bug fix
3. ✍️ Write your code and add tests
4. 📬 Submit a pull request
