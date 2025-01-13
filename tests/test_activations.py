import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import matplotlib.pyplot as plt
from fiztorch.tensor import Tensor
from fiztorch.nn.functional import sigmoid, tanh, relu, softmax, cross_entropy

# Create a large range Tensor
x_data = np.linspace(-10, 10, 1000)
x = Tensor(x_data, requires_grad=True)

# Prepare placeholders for outputs and gradients
activations = ["ReLU", "Sigmoid", "Tanh"]
outputs = []
grads = []

# Compute each activation and its gradient
for activation in activations:
    if activation == "ReLU":
        result = relu(x)
    elif activation == "Sigmoid":
        result = sigmoid(x)
    elif activation == "Tanh":
        result = tanh(x)
    
    # Backward with dummy gradient
    result.backward(Tensor(np.ones_like(result.data)))
    
    # Collect data
    outputs.append(result.data)
    grads.append(x.grad.data.copy())
    
    # Reset gradients for next activation
    x.zero_grad()

# Visualization
fig, axs = plt.subplots(3, 2, figsize=(12, 18))
fig.suptitle("Activation Functions and Their Gradients", fontsize=16)

for i, activation in enumerate(activations):
    # Plot Activation Output
    axs[i, 0].plot(x.data, outputs[i], label=f"{activation} Output")
    axs[i, 0].set_title(f"{activation} Activation")
    axs[i, 0].legend()
    
    # Plot Activation Gradient
    axs[i, 1].plot(x.data, grads[i], label=f"{activation} Gradient", color="orange")
    axs[i, 1].set_title(f"{activation} Gradient")
    axs[i, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()
