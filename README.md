# FizTorch: A Toy Tensor Library for Machine Learning

![Logo](assets/fiztorch.png)

## Introduction ✨
FizTorch is a toy implementation of a tensor library inspired by popular machine learning frameworks like PyTorch and TensorFlow. This project aims to create a basic tensor class with essential operations and functionality, helping you understand the core concepts behind tensor-based machine learning.

## Key Features 🔑
- ➕ Basic tensor operations: addition, subtraction, multiplication, division
- 🧮 Matrix multiplication
- 📏 Handling of tensor shapes and sizes
- 🔄 Gradient tracking and backpropagation
- 💾 Serialization and persistence
- 🧪 Unit testing and comprehensive documentation
- 🧠 Neural network modules and layers
- 📊 Data loading and batching
- 🔧 Optimization algorithms

## Examples 🚀

### Basic Tensor Operations

```python
from ftensor import FTensor as ft

# Creating higher-dimensional tensors
a = ft([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])  # Shape: (2, 2, 2)
b = ft([[[1, 0], [0, 1]], [[1, 1], [1, 1]]])  # Shape: (2, 2, 2)

# Basic operations
print("a + b:\n", a + b)                          # Element-wise addition
print("a - b:\n", a - b)                          # Element-wise subtraction
print("a * b:\n", a * b)                          # Element-wise multiplication

# Dot product
result = a.dot(b)
print("a.dot(b):\n", result)                      # Dot product

# More operations...
```

### Neural Network Training

```python
import numpy as np
from ftensor import nn, optim, data, utils

# Create a simple dataset
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, (1000, 1))
dataset = data.Dataset(list(zip(X, y)))
dataloader = data.DataLoader(dataset, batch_size=32)

# Create a model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
loss_fn = lambda pred, target: ((pred - target) ** 2).mean()
optimizer = optim.Adam(model.parameters(), learning_rate=0.001)

# Train the model
utils.train(model, dataloader, loss_fn, optimizer, epochs=10)
print("Training completed!")
```

## Implementation Goals 🎯

### Completed Functionalities ✅

- ✅ **Higher-Dimensional Tensors**: Support for tensors of arbitrary dimensions.
- ✅ **Basic Operations**: Element-wise addition, subtraction, and multiplication.
- ✅ **Dot Product**: Implemented dot product functionality for tensor operations.
- ✅ **Flattening and Transposing**: Methods to flatten and transpose tensors.
- ✅ **Element-wise Operations**: Logarithm, exponential, softmax, and ReLU derivative.
- ✅ **Reshaping Tensors**: Reshape tensors to desired dimensions.
- ✅ **Advanced Tensor Manipulations**: Tensor summation over specified axes.
- ✅ **Neural Network Modules**: Implemented Linear, ReLU, and Sigmoid layers.
- ✅ **Optimizers**: Adam optimizer for parameter updates.
- ✅ **Data Handling**: Dataset and DataLoader classes for batch processing.
- ✅ **Training Utility**: Helper function for model training.

### Known Issues ⚠️

- ⚠️ **Broadcasting Support**: Some broadcasting operations require further debugging.

### Future Additions and Features 🚀

- 🔲 **Additional Neural Network Layers**: Convolutional, pooling, and recurrent layers.
- 🔲 **More Optimizers**: Implement SGD, RMSprop, and other optimization algorithms.
- 🔲 **Support for Sparse Tensors**: Enhance functionality to handle sparse tensor representations.
- 🔲 **GPU Acceleration**: Integrate support for GPU computations for performance improvement.
- 🔲 **Comprehensive Documentation**: Provide detailed usage examples and API documentation.
- 🔲 **Expanded Unit Testing**: Cover more edge cases and functionality.
- 🔲 **Performance Benchmarks**: Create benchmarks to evaluate performance against other frameworks.

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

## License
[MIT License](LICENSE)