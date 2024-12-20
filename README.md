
# FizTorch: A Toy Tensor Library for Machine Learning

![Logo](assets/fiztorch.png)
[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://travis-ci.org/yourusername/FizTorch)


FizTorch is a lightweight deep learning framework designed for educational purposes and small-scale projects. It provides a simple and intuitive API for building and training neural networks, inspired by popular frameworks like PyTorch.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Tensor Operations**: Basic tensor operations with support for automatic differentiation.
- **Neural Network Layers**: Common neural network layers such as Linear and ReLU.
- **Sequential Model**: Easy-to-use sequential model for stacking layers.
- **Functional API**: Functional operations for common neural network functions.

## Installation

To install FizTorch, follow these steps:

1. **Clone the Repository**:
   ```sh
   git clone https://github.com/ahammadnafiz/FizTorch.git
   cd FizTorch
   ```

2. **Set Up a Virtual Environment** (optional but recommended):
   ```sh
   python -m venv fiztorch-env
   source fiztorch-env/bin/activate  # On Windows, use `fiztorch-env\Scripts\activate`
   ```

3. **Install Dependencies**:
   ```sh
   pip install -r requirements.txt
   ```

4. **Install FizTorch**:
   ```sh
   pip install -e .
   ```

## Usage

Here is a simple example of how to use FizTorch to build and train a neural network:

```python
import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F

# Define a simple neural network
model = Sequential(
    Linear(2, 3),
    ReLU(),
    Linear(3, 1)
)

# Create some input data
input = Tensor([[1.0, 2.0]], requires_grad=True)

# Forward pass
output = model(input)

# Backward pass
output.backward()

# Print the gradients
print(input.grad)
```

## Examples

## MNIST HAND DIGIT TEST
Neural network training on MNIST digits using a FizTorch library with Adam optimizer (configurable learning rate), batch support, real-time accuracy/loss tracking
![Training Process](assets/training_progress.gif)


### Linear Layer

```python
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear

# Create a linear layer
layer = Linear(2, 3)

# Create some input data
input = Tensor([[1.0, 2.0]])

# Forward pass
output = layer(input)

# Print the output
print(output)
```

### ReLU Activation

```python
from fiztorch.tensor import Tensor
from fiztorch.nn import ReLU

# Create a ReLU activation
relu = ReLU()

# Create some input data
input = Tensor([-1.0, 0.0, 1.0])

# Forward pass
output = relu(input)

# Print the output
print(output)
```

### Sequential Model

```python
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential

# Define a sequential model
model = Sequential(
    Linear(2, 3),
    ReLU(),
    Linear(3, 1)
)

# Create some input data
input = Tensor([[1.0, 2.0]])

# Forward pass
output = model(input)

# Print the output
print(output)
```

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Commit your changes (`git commit -am 'Add new feature'`).
4. Push to the branch (`git push origin feature-branch`).
5. Create a new Pull Request.

## License

FizTorch is licensed under the MIT License. See the [LICENSE](https://github.com/ahammadnafiz/FizTorch/blob/main/LICENSE) file for more information.

## Contact

For any questions or feedback, please open an issue or contact the maintainers.

---

Made with ❤️ by [ahammadnafiz](https://github.com/ahammadnafiz)
