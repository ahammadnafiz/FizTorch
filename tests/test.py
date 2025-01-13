import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

from fiztorch.tensor import Tensor
from fiztorch.nn.layers import Linear, ReLU
from fiztorch.nn.sequential import Sequential

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

output.foward()

# Backward pass
output.backward()

# Print the gradients
print(input.grad)