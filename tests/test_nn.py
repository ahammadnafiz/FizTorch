import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F

def test_linear_layer():
    layer = Linear(2, 3)
    input = Tensor([[1.0, 2.0]])
    output = layer(input)
    assert output.shape == (1, 3)

def test_relu():
    relu = ReLU()
    input = Tensor([-1.0, 0.0, 1.0])
    output = relu(input)
    assert np.array_equal(output.data, np.array([0.0, 0.0, 1.0]))

def test_sequential():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1)
    )
    input = Tensor([[1.0, 2.0]])
    output = model(input)
    assert output.shape == (1, 1)

def test_functional():
    input = Tensor([-1.0, 0.0, 1.0])

    # Test ReLU
    output = F.relu(input)
    assert np.array_equal(output.data, np.array([0.0, 0.0, 1.0]))

    # Test sigmoid
    output = F.sigmoid(input)
    assert np.all((output.data >= 0) & (output.data <= 1))

    # Test softmax
    output = F.softmax(input)
    assert np.allclose(np.sum(output.data), 1.0)

if __name__ == "__main__":
    pytest.main([__file__])