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
Test 4: 3D + 1D
FTensor result:
FTensor(shape=(2, 2, 2), dtype=list)
Data:
[[11, 22] [13, 24]]
[[15, 26] [17, 28]]
NumPy result:
[[[11 22]
  [13 24]]

 [[15 26]
  [17 28]]]
Test passed: Results match!

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
