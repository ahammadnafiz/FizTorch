import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import time
from fiztorch import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
from fiztorch.optim.optimizer import SGD
import fiztorch.nn.functional as F

model = Sequential(
    Linear(2, 64),  # Input layer (2 features to 64 neurons)
    ReLU(),
    Linear(64, 128),  # Hidden layer (64 neurons to 128 neurons)
    ReLU(),
    Linear(128, 64),  # Hidden layer (128 neurons to 64 neurons)
    ReLU(),
    Linear(64, 32),   # Hidden layer (64 neurons to 32 neurons)
    ReLU(),
    Linear(32, 1)     # Output layer (32 neurons to 1 output)
)

# Log function
def log_parameters_and_gradients(epoch, loss, model):
    """
    Logs parameters and loss after each epoch.

    Args:
        epoch (int): Current epoch number.
        loss (Tensor): Loss tensor.
        model (Sequential): The neural network model.
    """
    print(f"\n=== Epoch {epoch + 1} ===")
    print(f"Loss: {loss.data}")
    print("Updated Parameters:")
    for i, param in enumerate(model.parameters()):
        param_name = f"Parameter {i + 1} (Weight of Layer {i // 2 + 1})" if i % 2 == 0 else f"Parameter {i + 1} (Bias of Layer {i // 2 + 1})"
        print(f"\n{param_name}:")
        print(f"Value:\n{param.data}")

def train_example(delay=1.0):
    """
    Trains the model with a delay between epochs.

    Args:
        delay (float): Time in seconds to wait between each epoch's output.
    """
    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=0.01)

    # Generate a larger dummy dataset for demonstration
    np.random.seed(42)  # For reproducibility
    X_train = Tensor(np.random.rand(1000, 2), requires_grad=True)  # 1000 samples, 2 features
    y_train = Tensor(np.random.rand(1000, 1))  # 1000 target values

    # Training loop for 20 epochs
    for epoch in range(20):
        optimizer.zero_grad()  # Reset gradients
        predictions = model(X_train)  # Forward pass
        loss = F.mse_loss(predictions, y_train)  # Mean squared error loss
        loss.backward()  # Backpropagation
        optimizer.step()  # Update weights

        # Log parameters and loss
        log_parameters_and_gradients(epoch, loss, model)

        # Introduce a delay
        time.sleep(delay)

if __name__ == "__main__":
    # Train with a delay of 2 seconds between each epoch
    train_example(delay=2.0)
