# Project Overview

This document provides a high-level overview of the FizTorch project, its purpose, and its main components.

FizTorch is a lightweight deep learning framework designed for educational purposes and small-scale projects. It provides a simple and intuitive API for building and training neural networks, inspired by popular frameworks like PyTorch.

The project is organized into two main directories:

- **`fiztorch`**: This directory contains the core deep learning framework. It includes implementations of tensors, neural network layers, optimizers, and other essential components.
- **`fiznet`**: This directory contains higher-level models and utilities built on top of `fiztorch`. It provides convenient wrappers and abstractions for common machine learning tasks.

## The `fiztorch` Directory

The `fiztorch` directory is the heart of the framework, providing the fundamental building blocks for deep learning.

- **`tensor.py`**: This file defines the `Tensor` class, which is the primary data structure in FizTorch. It supports basic tensor operations and automatic differentiation, crucial for training neural networks.
- **`nn/`**: This subdirectory houses the components for building neural networks:
    - **`layers.py`**: Contains implementations of various neural network layers, such as `Linear`, `ReLU`, and `Sigmoid`. These are the basic units for constructing network architectures.
    - **`module.py`**: Defines a base `Module` class, which is a fundamental concept for encapsulating parts of a neural network. Layers and models inherit from this class.
    - **`sequential.py`**: Provides a `Sequential` container to easily stack layers in a feedforward manner.
    - **`activations.py`**: Implements various activation functions that introduce non-linearities into the network.
    - **`losses.py`**: Contains common loss functions like Binary Cross Entropy, used to evaluate the performance of the model during training.
    - **`functional.py`**: Offers a functional API for operations commonly used in neural networks, providing flexibility in how models are defined.
    - **`init_functions.py`**: Includes functions for initializing the weights of neural network layers.
- **`optim/`**: This subdirectory contains optimization algorithms:
    - **`optimizer.py`**: Defines base `Optimizer` classes and specific optimizers like `Adam` and `SGD`. Optimizers are used to update the model's parameters (weights) during training to minimize the loss function.
- **`utils/`**: This subdirectory provides utility functions:
    - **`broadcast.py`**: Contains functions related to tensor broadcasting, enabling operations on tensors of different shapes.
    - **`data.py`**: Includes utilities for handling datasets, such as creating data loaders.
    - **`visual.py`**: Provides tools for visualizing data or model performance (though it might be basic in this lightweight framework).

## The `fiznet` Directory

The `fiznet` directory builds upon `fiztorch` to offer more specialized and higher-level abstractions for common machine learning models and tasks. It aims to simplify the process of building and training models for specific use cases.

- **`base_model.py`**: Likely defines a base class for models within `fiznet`, providing a common structure and potentially shared functionalities.
- **`binarymodel.py`**: Suggests an implementation of a model specifically designed for binary classification tasks.
- **`neural_network.py`**: This file probably contains a general-purpose neural network class, possibly with more built-in functionalities (like training loops or evaluation metrics) compared to the barebones `Sequential` model in `fiztorch.nn`.
- **`cnn.py`**: Indicates an implementation of Convolutional Neural Networks (CNNs), which are specialized for tasks like image processing. This would be a significant feature built on `fiztorch`.
- **`knn.py`**: Suggests an implementation of the K-Nearest Neighbors algorithm, which is a non-parametric machine learning model. This shows that `fiznet` might not be limited to just neural network models.
- **`utils/`**: Similar to `fiztorch`, this subdirectory likely contains utility functions, but these would be specific to the needs of the `fiznet` models and abstractions.
    - **`tools.py`**: A generic name for a file containing various helper functions or tools for `fiznet`.

In essence, `fiznet` acts as a companion library to `fiztorch`, providing pre-built models and tools that leverage the core `fiztorch` framework to accelerate development for common machine learning problems.

## How the Project Works: A Summary

FizTorch, along with its higher-level companion `fiznet`, provides a comprehensive environment for building and training machine learning models, particularly neural networks. Here's a general workflow:

1.  **Tensor Creation**: Data is represented using `fiztorch.tensor.Tensor` objects. These tensors are similar to NumPy arrays but have the added capability of tracking gradients for automatic differentiation, which is essential for training neural networks.

2.  **Model Definition**:
    *   **Using `fiztorch.nn`**: You can build models by combining layers from `fiztorch.nn.layers` (e.g., `Linear`, `ReLU`, `Sigmoid`). The `fiztorch.nn.sequential.Sequential` container is often used to stack these layers in a straightforward manner. This approach gives you fine-grained control over the network architecture.
    *   **Using `fiznet`**: For common tasks, `fiznet` might offer pre-built model classes (e.g., `fiznet.neural_network.NeuralNetwork`, `fiznet.cnn.CNN`, or `fiznet.binarymodel.BinaryModel`). These classes encapsulate more of the model structure and potentially some training logic.

3.  **Data Handling**:
    *   Input data (features and labels) is converted into `Tensor` objects.
    *   `fiztorch.utils.data` might provide utilities for creating datasets and data loaders to feed data to the model in batches during training, though the `README.md` example shows manual batching with NumPy.

4.  **Loss Function Selection**: A loss function is chosen from `fiztorch.nn.losses` (e.g., `F.binary_cross_entropy` for binary classification) to measure the discrepancy between the model's predictions and the actual target values.

5.  **Optimizer Selection**: An optimizer is chosen from `fiztorch.optim.optimizer` (e.g., `Adam`, `SGD`). The optimizer's role is to update the model's parameters (weights and biases) based on the gradients computed during backpropagation to minimize the loss.

6.  **Training Loop**:
    *   The model is set to training mode.
    *   For a number of epochs (iterations over the entire dataset):
        *   Iterate over the training data, typically in batches.
        *   **Forward Pass**: Input data is passed through the model to get predictions.
        *   **Loss Calculation**: The loss function computes the loss between predictions and true labels.
        *   **Backward Pass (Backpropagation)**: The `loss.backward()` method is called. This automatically computes the gradients of the loss with respect to all model parameters that have `requires_grad=True`.
        *   **Optimizer Step**: The `optimizer.step()` method updates the model parameters using the computed gradients.
        *   The optimizer's gradients are typically reset at the beginning of each batch using `optimizer.zero_grad()`.

7.  **Evaluation**: After training (or periodically during training), the model's performance is evaluated on a separate test dataset using appropriate metrics (e.g., accuracy, precision, recall, confusion matrix, as shown in the `README.md` example).

8.  **Prediction**: Once trained, the model can be used to make predictions on new, unseen data.

The examples provided in the main `README.md` (like the breast cancer classification example) and in the `examples/` directory (e.g., `mnist_example.py`, `regression_example.py`) serve as practical illustrations of this workflow. They demonstrate how to load data, create and train models, and evaluate their performance using FizTorch.

For instance, the `README.md` example for breast cancer classification clearly shows:
- Loading and preprocessing data.
- Creating a `Sequential` model with `Linear` and `Sigmoid` layers.
- Using `F.binary_cross_entropy` as the loss function.
- Employing the `Adam` optimizer.
- A manual training loop performing forward pass, loss calculation, backward pass, and optimizer step.
- Evaluation using `classification_report` and `confusion_matrix` from scikit-learn.

This workflow is characteristic of most modern deep learning frameworks, and FizTorch aims to provide a simplified yet functional version of it.
