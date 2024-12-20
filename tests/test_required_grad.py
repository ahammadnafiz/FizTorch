import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F
from fiztorch.optim.optimizer import SGD

# Initialize model parameters
W = Tensor(np.random.randn(2, 3), requires_grad=True)
b = Tensor(np.zeros(3), requires_grad=True)

# Dummy dataset
X = Tensor(np.random.randn(5, 2))  # Input
y = Tensor(np.array([0, 1, 2, 1, 0]))  # Target

# Training loop
optimizer = SGD([W, b], lr=0.1)
for epoch in range(50):
    # Forward pass
    logits = X @ W + b
    loss = F.cross_entropy(logits, y)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Log loss
    print(f"Epoch {epoch+1}, Loss: {loss.data}")
