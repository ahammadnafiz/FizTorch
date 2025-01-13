import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from fiztorch.nn.functional import sigmoid, tanh, softmax, relu, Tensor

input_tensor = Tensor([[-1.0, 0.0, 1.0], [2.0, -2.0, 0.5]], requires_grad=True)
relu_output = relu(input_tensor)

# Perform a backward pass with a gradient of ones
relu_output.backward(Tensor(np.ones_like(relu_output.data)))

print("Input Gradients:")
print(input_tensor.grad.data)

sigmoid_output = sigmoid(input_tensor)
sigmoid_output.backward(Tensor(np.ones_like(sigmoid_output.data)))
print("Sigmoid Output:")
print(sigmoid_output.data)
print("Sigmoid Input Gradients:")
print(input_tensor.grad.data)

tanh_output = tanh(input_tensor)
tanh_output.backward(Tensor(np.ones_like(tanh_output.data)))
print("Tanh Output:")
print(tanh_output.data)
print("Tanh Input Gradients:")
print(input_tensor.grad.data)

softmax_output = softmax(input_tensor)
softmax_output.backward(Tensor(np.ones_like(softmax_output.data)))
print("Softmax Output:")
print(softmax_output.data)
print("Softmax Input Gradients:")
print(input_tensor.grad.data)