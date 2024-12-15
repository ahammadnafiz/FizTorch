
# FizTorch: A Toy Tensor Library for Machine Learning

![Logo](assets/fiztorch.png)

## Introduction ✨
FizTorch is a lightweight tensor library built to mimic core functionalities of machine learning frameworks like PyTorch and TensorFlow. Designed as an educational project, it provides a hands-on way to explore and understand tensor operations, gradient tracking, and basic neural network constructs.

## Key Features 🔑
- ➕ **Basic Tensor Operations:** Element-wise addition, subtraction, multiplication, and division.
- 🔢 **Matrix Operations:** Support for dot products and matrix multiplication.
- 🔹 **Gradient Tracking:** Automated gradient computation for backpropagation.
- 🔄 **Shape Manipulation:** Reshaping, flattening, and transposing tensors.
- 🗂 **Data Utilities:** Dataset and DataLoader for batch processing.
- 🔧 **Optimization:** Adam optimizer for training.
- 🔖 **Neural Network Modules:** Linear layers, activation functions, and sequential models.
- 📊 **Testing and Debugging:** Built-in unit tests and test cases for key functionalities.

## Test-Driven Examples 🚀
### **1. Gradient Tracking for Basic Tensor Operations**
```python
from ftensor import FTensor as ft

# Initialize tensors
x = ft([1.0])
y = ft([2.0])
z = ft([3.0])

# Perform operations
a = x + y
b = a * z
c = b.relu()
d = c.sum()

d.backward()  # Compute gradients

# Display gradients
print("Gradient of x:", x.grad)
print("Gradient of y:", y.grad)
print("Gradient of z:", z.grad)
```
**Output:**
```
Gradient of x: 3.0
Gradient of y: 3.0
Gradient of z: 3.0
```

## Core Functionalities 🎯
### **Completed Features**
- ✅ Tensor Operations: Addition, subtraction, multiplication, division.
- ✅ Matrix Multiplication: Dot products and batched operations.
- ✅ Gradient Tracking: Automatic differentiation with `.backward()`.
- ✅ Shape Manipulations: Reshape, flatten, and transpose.
- ✅ Neural Network Modules: Linear layers, ReLU, Sigmoid.
- ✅ Optimization: Adam optimizer for training.
- ✅ Data Handling: Dataset and DataLoader for efficient batch processing.
- ✅ Unit Testing: Comprehensive test cases for all key functionalities.

### **Known Limitations**
- ⚠️ Limited Broadcasting: Needs enhancement for full broadcasting support.
- ⚠️ GPU Acceleration: Currently CPU-bound.

### **Planned Features**
- □ Advanced Layers: Convolutional, pooling, and recurrent layers.
- □ Additional Optimizers: Support for SGD, RMSprop, and others.
- □ Sparse Tensors: Improved handling of sparse data structures.
- □ GPU Support: Integration with CUDA for faster computation.
- □ Neural Network Training: Full end-to-end pipeline for training models.
- □ Documentation: Expanded guides and tutorials.

## Getting Started 🚀
1. Clone the repository:
   ```bash
   git clone https://github.com/ahammadnafiz/FizTorch.git
   cd FizTorch
   ```
2. Set up your environment:
   - Install Python 3.x
   - Install dependencies using `pip install -r requirements.txt`

## Contributing 🤝
We welcome contributions to enhance FizTorch! Follow these steps:
1. Fork the repository.
2. Create a feature branch.
3. Implement changes and add tests.
4. Submit a pull request for review.

## License 🔒
FizTorch is licensed under the [MIT License](LICENSE).
