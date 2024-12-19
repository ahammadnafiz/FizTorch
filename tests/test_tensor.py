import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from fiztorch.tensor import Tensor

def test_tensor_creation():
    # Test various ways of creating tensors
    data = [[1, 2], [3, 4]]
    t1 = Tensor(data)
    assert np.array_equal(t1.data, np.array(data))
    
    t2 = Tensor(5.0)
    assert t2.data == 5.0
    
    t3 = Tensor(np.array([1, 2, 3]))
    assert np.array_equal(t3.data, np.array([1, 2, 3]))

def test_tensor_operations():
    t1 = Tensor([1, 2, 3], requires_grad=True)
    t2 = Tensor([4, 5, 6], requires_grad=True)
    
    # Test addition
    result = t1 + t2
    assert np.array_equal(result.data, np.array([5, 7, 9]))
    
    # Test multiplication
    result = t1 * t2
    assert np.array_equal(result.data, np.array([4, 10, 18]))

def test_backward_propagation():
    # Test simple backward pass
    x = Tensor(2.0, requires_grad=True)
    y = x * x
    y.backward()
    assert x.grad.data == 4.0
    
    # Test more complex backward pass
    x = Tensor(2.0, requires_grad=True)
    y = Tensor(3.0, requires_grad=True)
    z = x * y + y
    z.backward()
    assert x.grad.data == 3.0
    assert y.grad.data == 3.0
    
if __name__ == "__main__":
    pytest.main([__file__])