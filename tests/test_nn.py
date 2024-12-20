import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F

def test_linear_layer():
    # Standard test
    layer = Linear(2, 3)
    input = Tensor([[1.0, 2.0]])
    output = layer(input)
    assert output.shape == (1, 3), f"Expected shape (1, 3), got {output.shape}"

    # Edge case: Empty tensor
    input = Tensor([])
    with pytest.raises(ValueError):
        layer(input)

    # Edge case: Incorrect input shape
    input = Tensor([1.0])  # Not a 2D tensor
    with pytest.raises(ValueError):
        layer(input)

def test_relu():
    relu = ReLU()

    # Standard test
    input = Tensor([-1.0, 0.0, 1.0])
    output = relu(input)
    assert np.array_equal(output.data, np.array([0.0, 0.0, 1.0])), \
        f"ReLU output mismatch: {output.data}"

    # Edge case: Large positive and negative values
    input = Tensor([-1e6, 1e6])
    output = relu(input)
    assert np.array_equal(output.data, np.array([0.0, 1e6])), \
        f"ReLU failed with large values: {output.data}"

def test_sequential():
    model = Sequential(
        Linear(2, 3),
        ReLU(),
        Linear(3, 1)
    )
    input = Tensor([[1.0, 2.0]])
    output = model(input)
    assert output.shape == (1, 1), f"Expected shape (1, 1), got {output.shape}"

    # Edge case: Empty input
    input = Tensor([])
    with pytest.raises(ValueError):
        model(input)

def test_functional():
    input = Tensor([-1.0, 0.0, 1.0])

    # Test ReLU
    output = F.relu(input)
    assert np.array_equal(output.data, np.array([0.0, 0.0, 1.0])), \
        f"Functional ReLU output mismatch: {output.data}"

    # Test sigmoid
    output = F.sigmoid(input)
    expected = 1 / (1 + np.exp(-input.data))
    assert np.allclose(output.data, expected, atol=1e-6), \
        f"Sigmoid output mismatch: {output.data} vs {expected}"

    # Test softmax
    output = F.softmax(input)
    expected = np.exp(input.data) / np.sum(np.exp(input.data))
    assert np.allclose(output.data, expected, atol=1e-6), \
        f"Softmax output mismatch: {output.data} vs {expected}"
    assert np.isclose(np.sum(output.data), 1.0, atol=1e-6), \
        f"Softmax probabilities do not sum to 1: {np.sum(output.data)}"

if __name__ == "__main__":
    pytest.main([__file__])