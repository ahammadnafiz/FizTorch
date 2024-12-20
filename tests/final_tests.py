import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from fiztorch import Tensor
from fiztorch.optim.optimizer import SGD

def test_tensor_creation():
    # Test different data types
    data_types = [
        [[1, 2], [3, 4]],  # list of lists
        np.array([[1, 2], [3, 4]]),  # numpy array
        2.5,  # scalar
        [1, 2, 3]  # 1D list
    ]
    
    for data in data_types:
        t = Tensor(data)
        assert isinstance(t.data, np.ndarray)
        if isinstance(data, (list, np.ndarray)):
            assert np.array_equal(t.data, np.array(data))
        else:
            assert t.data == data

def test_tensor_properties():
    t = Tensor([[1, 2, 3], [4, 5, 6]])
    assert t.shape == (2, 3)
    assert np.array_equal(t.T.data, np.array([[1, 4], [2, 5], [3, 6]]))
    assert t.requires_grad == False
    
    t_grad = Tensor([[1, 2], [3, 4]], requires_grad=True)
    assert t_grad.requires_grad == True
    assert t_grad.is_leaf == True

def test_tensor_arithmetic():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([[5, 6], [7, 8]], requires_grad=True)
    
    # Addition
    z = x + y
    assert np.array_equal(z.data, np.array([[6, 8], [10, 12]]))
    
    # Subtraction
    z = x - y
    assert np.array_equal(z.data, np.array([[-4, -4], [-4, -4]]))
    
    # Multiplication
    z = x * y
    assert np.array_equal(z.data, np.array([[5, 12], [21, 32]]))
    
    # Division
    z = x / 2
    assert np.array_equal(z.data, np.array([[0.5, 1], [1.5, 2]]))

def test_matmul():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([[5, 6], [7, 8]], requires_grad=True)
    z = x @ y
    assert np.array_equal(z.data, np.array([[19, 22], [43, 50]]))

def test_broadcasting():
    # Broadcasting scalar
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor(2, requires_grad=True)
    z = x + y
    assert np.array_equal(z.data, np.array([[3, 4], [5, 6]]))
    
    # Broadcasting vectors
    x = Tensor([[1], [2]], requires_grad=True)
    y = Tensor([[3, 4]], requires_grad=True)
    z = x + y
    assert np.array_equal(z.data, np.array([[4, 5], [5, 6]]))

def test_broadcasting_gradients():
    # Test gradient computation with broadcasting
    x = Tensor([[1], [2]], requires_grad=True)  # 2x1 matrix
    y = Tensor([[3, 4, 5]], requires_grad=True)  # 1x3 matrix
    z = x + y  # Broadcasting to 2x3
    z.backward()
    
    # Check shapes
    assert x.grad.shape == (2, 1)
    assert y.grad.shape == (1, 3)
    
    # Check values
    assert np.array_equal(x.grad.data, np.array([[3], [3]]))  # Sum across broadcast dimension
    assert np.array_equal(y.grad.data, np.array([[2, 2, 2]]))  # Sum across broadcast dimension

def test_complex_operations():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    y = Tensor([[5, 6], [7, 8]], requires_grad=True)
    
    # Complex computation: f = (x + y) * (x * y)
    z = (x + y) * (x * y)
    z.backward()
    
    assert x.grad is not None
    assert y.grad is not None

def test_reshape():
    x = Tensor([[1, 2, 3], [4, 5, 6]], requires_grad=True)
    y = x.reshape(3, 2)
    assert y.shape == (3, 2)
    y.backward()
    assert x.grad.shape == (2, 3)

def test_sum_and_mean():
    x = Tensor([[1, 2], [3, 4]], requires_grad=True)
    
    # Test sum
    y = x.sum()
    assert y.data == 10
    y.backward()
    assert np.array_equal(x.grad.data, np.ones_like(x.data))
    
    # Test mean
    x.grad = None  # Reset gradient
    y = x.mean()
    assert y.data == 2.5
    y.backward()
    assert np.array_equal(x.grad.data, np.ones_like(x.data) * 0.25)

def test_power():
    x = Tensor([2, 3], requires_grad=True)
    y = x ** 2
    y.backward()
    assert np.array_equal(y.data, np.array([4, 9]))
    assert np.array_equal(x.grad.data, np.array([4, 6]))

if __name__ == "__main__":
    pytest.main([__file__])