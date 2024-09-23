![Logo](assets/fiztorch.png)

## Introduction
This is a toy implementation of a Tensor class, inspired by popular machine learning libraries like PyTorch and TensorFlow. The goal of this project is to create a basic tensor class with essential operations and functionality, to help you understand the core concepts behind tensor-based machine learning.

## Key Features
- Basic tensor operations: addition, subtraction, multiplication, division
- Matrix multiplication
- Handling of tensor shapes and sizes
- Gradient tracking and backpropagation
- Serialization and persistence
- Unit testing and comprehensive documentation

## Project Roadmap

### Basic Functionality
1. **Implement Basic Operations** ✨
   - [x] Addition (`__add__`)
   - [x] Subtraction (`__sub__`)
   - [x] Multiplication (`__mul__`)
   - [x] Division (`__truediv__`)
2. **Implement Matrix Multiplication** 🔢
   - [ ] Create `matmul` method
3. **Shape and Size Handling** 🔍
   - [x] Enhance the `shape` method for higher-dimensional tensors
4. **Gradient Tracking** 🧠
   - [ ] Implement `backward` method for gradient accumulation
5. **String Representation** 🔍
   - [x] Implement `__repr__` for better tensor visualization

### Intermediate Features
6. **Support Higher-Dimensional Tensors** ⚙️
   - [x] Modify operations to handle tensors with more than two dimensions
7. **Implement Advanced Operations** 🧠
   - [x] Element-wise power operation
   - [ ] Reduction operations (sum, mean, max, min)
   - [x] Transpose method
8. **Automatic Differentiation** 🧠
   - [ ] Develop a system to build a computational graph for backpropagation

### Performance Enhancements
9. **Integrate with NumPy (Optional)** 🚀
   - [ ] Use NumPy for performance optimization in operations
10. **Implement Optimizers** 🧠
    - [ ] Create basic optimizers (SGD, Adam)

### Persistence and Usability
11. **Implement Serialization** 💾
    - [ ] Add methods for saving/loading tensors to/from disk
12. **Enhanced Gradient Management** 🧠
    - [ ] Create methods for resetting and clipping gradients

### Testing and Documentation
13. **Set Up Unit Tests** ✅
    - [ ] Write unit tests for all methods to ensure correctness
14. **Create Comprehensive Documentation** 📚
    - [ ] Document the API, including examples and use cases
15. **User Interface Improvements** 🚀
    - [ ] Consider operator overloading for easier syntax

### Future Enhancements (Post MVP)
16. **Explore Additional Features** 🔍
    - [ ] Look into more advanced tensor operations
    - [ ] Investigate compatibility with other machine learning frameworks

## Getting Started
To get started, clone the repository and set up your development environment. You'll need Python 3.x installed on your system.

```
git clone https://github.com/ahammadnafiz/FizTorch.git
cd tensor-class-project
```

## Contributing
Contributions are welcome! If you'd like to help, please follow these guidelines:

1. Fork the repository
2. Create a new branch for your feature or bug fix
3. Write your code and add tests
4. Submit a pull request
