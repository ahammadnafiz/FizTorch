import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import numpy as np
from typing import List, Tuple

from fiztorch.tensor import Tensor

def test_gradients():
    """Run comprehensive tests for gradient tracking"""
    # Test 1: Simple addition
    def test_addition():
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x + y
        z.backward()
        assert np.allclose(x.grad.data, [1.0]), "Addition gradient for x failed"
        assert np.allclose(y.grad.data, [1.0]), "Addition gradient for y failed"
        print("✓ Addition gradient test passed")

    # Test 2: Multiplication
    def test_multiplication():
        x = Tensor([2.0], requires_grad=True)
        y = Tensor([3.0], requires_grad=True)
        z = x * y
        z.backward()
        assert np.allclose(x.grad.data, [3.0]), "Multiplication gradient for x failed"
        assert np.allclose(y.grad.data, [2.0]), "Multiplication gradient for y failed"
        print("✓ Multiplication gradient test passed")

    # Test 3: Matrix multiplication
    def test_matmul():
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = Tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
        z = x @ y
        z.backward(Tensor(np.ones_like(z.data)))
        expected_x_grad = np.array([[11., 15.], [11., 15.]])
        expected_y_grad = np.array([[4., 4.], [6., 6.]])
        assert np.allclose(x.grad.data, expected_x_grad), "Matrix multiplication gradient for x failed"
        assert np.allclose(y.grad.data, expected_y_grad), "Matrix multiplication gradient for y failed"
        print("✓ Matrix multiplication gradient test passed")

    # Test 4: Broadcasting
    def test_broadcasting():
        x = Tensor([[1.0], [2.0]], requires_grad=True)
        y = Tensor([[3.0, 4.0]], requires_grad=True)
        z = x + y  # Broadcasting
        z.backward(Tensor(np.ones_like(z.data)))
        assert np.allclose(x.grad.data, [[2.0], [2.0]]), "Broadcasting gradient for x failed"
        assert np.allclose(y.grad.data, [[2.0, 2.0]]), "Broadcasting gradient for y failed"
        print("✓ Broadcasting gradient test passed")

    # Test 5: Power operation
    def test_power():
        x = Tensor([2.0], requires_grad=True)
        z = x ** 2
        z.backward()
        assert np.allclose(x.grad.data, [4.0]), "Power gradient failed"
        print("✓ Power operation gradient test passed")

    # Test 6: Complex computation graph
    def test_complex_graph():
        x = Tensor([1.0], requires_grad=True)
        y = Tensor([2.0], requires_grad=True)
        a = x * y
        b = a + y
        c = b ** 2
        c.backward()
        # dc/dx = 2(xy + y) * y
        # dc/dy = 2(xy + y) * (x + 1)
        expected_x_grad = 2 * (1 * 2 + 2) * 2
        expected_y_grad = 2 * (1 * 2 + 2) * (1 + 1)
        assert np.allclose(x.grad.data, [expected_x_grad]), "Complex graph gradient for x failed"
        assert np.allclose(y.grad.data, [expected_y_grad]), "Complex graph gradient for y failed"
        print("✓ Complex computation graph test passed")

    # Test 7: Sum operation
    def test_sum():
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.sum()
        y.backward()
        assert np.allclose(x.grad.data, np.ones_like(x.data)), "Sum gradient failed"
        print("✓ Sum operation gradient test passed")

    # Test 8: Mean operation
    def test_mean():
        x = Tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
        y = x.mean()
        y.backward()
        assert np.allclose(x.grad.data, np.ones_like(x.data) * 0.25), "Mean gradient failed"
        print("✓ Mean operation gradient test passed")

    # Run all tests
    tests = [
        test_addition,
        test_multiplication,
        test_matmul,
        test_broadcasting,
        test_power,
        test_complex_graph,
        test_sum,
        test_mean
    ]

    print("Starting gradient tests...\n")
    for test in tests:
        try:
            test()
        except AssertionError as e:
            print(f"❌ {test.__name__} failed: {str(e)}")
        except Exception as e:
            print(f"❌ {test.__name__} failed with unexpected error: {str(e)}")
    print("\nGradient tests completed.")

# Run the tests
test_gradients()