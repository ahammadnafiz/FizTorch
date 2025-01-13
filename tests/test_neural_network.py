import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from fiztorch import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
from fiztorch.optim.optimizer import SGD
import fiztorch.nn.functional as F

# Define a neural network
model = Sequential(
    Linear(2, 3),
    ReLU(),
    Linear(3, 1)
)

# Create some input data
input = Tensor([[1.0, 2.0]], requires_grad=True)

# Example of training a model
def train_example():
    optimizer = SGD(model.parameters(), lr=0.01)

    # Dummy data for demonstration
    X_train = Tensor(np.random.rand(100, 2), requires_grad=True)
    y_train = Tensor(np.random.rand(100, 1))

    for epoch in range(5):  # Simulate 5 epochs of training
        optimizer.zero_grad()
        predictions = model(X_train)
        loss = F.mse_loss(predictions, y_train)
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch+1}, Loss: {loss.data}")

if __name__ == "__main__":
    train_example()