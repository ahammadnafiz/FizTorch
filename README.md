
<div align="center">

![Logo](assets/fiztorch.png)

🔥 A Minimal Deep Learning Framework 🔥

[![Python Version](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://travis-ci.org/yourusername/FizTorch)
[![Stars](https://img.shields.io/github/stars/ahammadnafiz/FizTorch?style=social)](https://github.com/ahammadnafiz/FizTorch/stargazers)

📊 🔢 🧮 🤖 📈

</div>


FizTorch is a lightweight deep learning framework designed for educational purposes and small-scale projects. It provides a simple and intuitive API for building and training neural networks, inspired by popular frameworks like PyTorch.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)
- [Roadmap](#roadmap)

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
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

from fiztorch.tensor import Tensor
from fiztorch.nn.layers import Linear, ReLU, Sigmoid
from fiztorch.nn.sequential import Sequential
import fiztorch.nn.functional as F
import fiztorch.optim.optimizer as opt

def load_data():
    X, y = load_breast_cancer(return_X_y=True)
    X = StandardScaler().fit_transform(X)
    return train_test_split(Tensor(X), Tensor(y), test_size=0.2, random_state=42)

def create_model():
    return Sequential(Linear(30, 64), ReLU(), Linear(64, 32), ReLU(), Linear(32, 1), Sigmoid())

def train_epoch(model, optimizer, X_train, y_train, batch_size=32):
    indices = np.random.permutation(len(X_train.data))
    for i in range(0, len(X_train.data), batch_size):
        batch = indices[i:i+batch_size]
        optimizer.zero_grad()
        loss = F.binary_cross_entropy(model(Tensor(X_train.data[batch])), Tensor(y_train.data[batch]))
        loss.backward()
        optimizer.step()

def evaluate(model, X, y):
    preds = model(X).data > 0.5
    print(classification_report(y.data, preds))
    print(confusion_matrix(y.data, preds))

def main():
    X_train, X_test, y_train, y_test = load_data()
    model, optimizer = create_model(), opt.Adam(model.parameters(), lr=0.001)
    for _ in range(100): train_epoch(model, optimizer, X_train, y_train)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()

```

## Examples

### MNIST HAND DIGIT TEST
Neural network training on MNIST digits using  FizTorch library with Adam optimizer (configurable learning rate), batch support, real-time accuracy/loss tracking
![Training Process](assets/training_progress.gif)

### California Housing TEST
Neural network training on California Housing Dataset using  FizTorch library
![Training Process](assets/training_progress_regression.gif)


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
## TODO

- [x] Implement basic tensor operations
- [x] Add support for automatic differentiation
- [x] Create fundamental neural network layers
- [x] Build sequential model functionality
- [x] Implement basic optimizers
- [x] Add MNIST digit recognition example
- [x] Add California housing regression example
- [ ] Add more activation functions (Leaky ReLU, ELU, SELU)
- [ ] Implement convolutional layers
- [ ] Add batch normalization
- [ ] Support GPU acceleration
- [ ] Create comprehensive documentation
- [ ] Add unit tests
- [ ] Implement data loading utilities
- [ ] Add model saving/loading functionality
- [ ] Implement dropout layers
- [ ] Add learning rate schedulers
- [x] Create visualization utilities
- [ ] Support multi-GPU training
- [ ] Add model quantization

## Roadmap

### Phase 1: Core Features

- Enhance tensor operations with more advanced functionalities (e.g., broadcasting).
- Add support for GPU acceleration (e.g., via CUDA or ROCm).
- Improve the API for ease of use and consistency.

### Phase 2: Neural Network Enhancements

- Add additional layers such as Convolutional, Dropout, and BatchNorm.
- Expand activation functions (e.g.,ELU).
- Integrate pre-trained models for common tasks.

### Phase 3: Training and Optimization

- Implement additional optimizers
- Add learning rate schedulers.
- Enhance support for custom loss functions.

### Phase 4: Dataset and Data Loading

- Provide built-in dataset utilities (e.g., MNIST, CIFAR).
- Create a flexible data loader with augmentation capabilities.

### Phase 5: Visualization and Monitoring

- Add utilities for loss/accuracy visualization.
- Integrate real-time training monitoring (e.g., TensorBoard support).

### Phase 6: Community Contributions

- Establish guidelines for community-driven feature additions.
- Host challenges to encourage usage and development.

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

<div align="center">
Made with ❤️ by <a href="https://github.com/ahammadnafiz">ahammadnafiz</a>
</div>
