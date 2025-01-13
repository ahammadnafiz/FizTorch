import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from fiztorch.nn.layers import Linear, ReLU, Softmax
from fiztorch.nn.sequential import Sequential
from fiztorch.tensor import Tensor

# Define a fully connected neural network for a specific dataset
class FullyConnectedNN:
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Initialize a fully connected neural network.

        Args:
            input_dim (int): Number of input features.
            hidden_dim (int): Number of neurons in the hidden layer.
            output_dim (int): Number of output classes or regression targets.
        """
        self.model = Sequential(
            Linear(input_dim, hidden_dim),  # First layer
            ReLU(),                        # Activation for hidden layer
            Linear(hidden_dim, output_dim),  # Second layer
            Softmax(axis=1)                # Output layer with softmax
        )

    def forward(self, x):
        """
        Perform a forward pass through the network.

        Args:
            x (Tensor): Input tensor.

        Returns:
            Tensor: Output tensor.
        """
        return self.model(x)

    def parameters(self):
        """
        Get all parameters of the network.

        Returns:
            Iterator[Tensor]: Iterator over all parameters in the network.
        """
        return self.model.parameters()

    def zero_grad(self):
        """
        Reset all gradients to zero.
        """
        self.model.zero_grad()

# Example usage:
if __name__ == "__main__":
    input_dim = 10  # Number of features
    hidden_dim = 5  # Number of hidden neurons
    output_dim = 3  # Number of output classes

    # Initialize the network
    nn = FullyConnectedNN(input_dim, hidden_dim, output_dim)

    # Generate random input data
    batch_size = 4
    x = Tensor(np.random.rand(batch_size, input_dim), requires_grad=True)

    # Forward pass
    output = nn.forward(x)

    print("Output:", output)
