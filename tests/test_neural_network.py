import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from fiztorch.nn.layers import Linear, ReLU, Sigmoid, LeakyReLU
from fiztorch.nn.sequential import Sequential
from fiztorch.tensor import Tensor

if __name__ == "__main__":
    # Define dimensions
    input_dim = 5  # Number of input features
    hidden_dim = 10  # Number of neurons in the hidden layer
    output_dim = 3  # Number of output classes

    # Define a simple neural network
    model = Sequential(
        Linear(input_dim, hidden_dim),  # First layer
        LeakyReLU(negative_slope=0.1),  # LeakyReLU for hidden layer
        Linear(hidden_dim, output_dim),  # Second layer
        Sigmoid()                        # Output layer with Sigmoid
    )

    # Generate random input data
    batch_size = 2
    x = Tensor(np.random.rand(batch_size, input_dim), requires_grad=True)

    # Forward pass
    print("=== Forward Pass ===")
    output = model(x)
    print("Output:", output)

    # Compute dummy loss (mean of the output)
    loss = output.mean()
    print("\n=== Loss ===")
    print("Loss:", loss)

    # Backward pass: compute gradients
    print("\n=== Backward Pass ===")
    loss.backward()

    # Check gradients
    print("\n=== Gradients ===")
    for idx, param in enumerate(model.parameters()):
        print(f"Parameter {idx}:")
        print(f"Data:\n{param.data}")
        print(f"Gradient:\n{param.grad}\n")