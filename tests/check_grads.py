import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fiztorch import Tensor

def numerical_gradient(func, x, epsilon=1e-7):
    """
    Compute numerical gradient using finite differences.

    Args:
        func: Function that takes a Tensor and returns a scalar Tensor
        x: Input Tensor at which to evaluate gradient
        epsilon: Small perturbation for finite difference

    Returns:
        Numerical approximation of gradient with respect to x
    """
    grad = np.zeros_like(x.data)
    it = np.nditer(x.data, flags=['multi_index'])

    while not it.finished:
        idx = it.multi_index

        # Compute f(x + epsilon)
        old_value = x.data[idx]
        x.data[idx] = old_value + epsilon
        plus_epsilon = func(x).data

        # Compute f(x - epsilon)
        x.data[idx] = old_value - epsilon
        minus_epsilon = func(x).data

        # Restore original value
        x.data[idx] = old_value

        # Compute numerical gradient
        grad[idx] = (plus_epsilon - minus_epsilon) / (2 * epsilon)
        it.iternext()

    return grad

def check_gradients(func, x, rtol=1e-5, atol=1e-8):
    """
    Check if analytical gradients match numerical gradients.

    Args:
        func: Function that takes a Tensor and returns a scalar Tensor
        x: Input Tensor at which to check gradients
        rtol: Relative tolerance for comparison
        atol: Absolute tolerance for comparison

    Returns:
        bool: True if gradients match within tolerance
        float: Maximum relative error
        float: Maximum absolute error
    """
    # Ensure x requires gradients
    if not x.requires_grad:
        x = Tensor(x.data, requires_grad=True)

    # Compute analytical gradient
    y = func(x)
    y.backward()
    analytical_grad = x.grad.data

    # Compute numerical gradient
    numerical_grad = numerical_gradient(func, x)

    # Compare gradients
    rel_error = np.abs(analytical_grad - numerical_grad) / (np.abs(numerical_grad) + 1e-8)
    abs_error = np.abs(analytical_grad - numerical_grad)

    max_rel_error = np.max(rel_error)
    max_abs_error = np.max(abs_error)
    gradients_match = np.allclose(analytical_grad, numerical_grad, rtol=rtol, atol=atol)

    return gradients_match, max_rel_error, max_abs_error

def test_gradient_checker():
    """
    Test cases for gradient checking.
    """
    # Test case 1: Simple square function
    def square_func(x):
        return (x ** 2).sum()

    x = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=True)
    match, rel_err, abs_err = check_gradients(square_func, x)
    print(f"Square function gradient check: Match={match}, Rel err={rel_err}, Abs err={abs_err}")
    assert match, f"Square function gradient check failed. Rel err: {rel_err}, Abs err: {abs_err}"

    # Test case 2: Matrix multiplication
    def matmul_func(x):
        return (x @ Tensor([[1.0], [2.0], [3.0]])).sum()

    x = Tensor(np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), requires_grad=True)
    match, rel_err, abs_err = check_gradients(matmul_func, x)
    print(f"Matrix multiplication gradient check: Match={match}, Rel err={rel_err}, Abs err={abs_err}")
    assert match, f"Matrix multiplication gradient check failed. Rel err: {rel_err}, Abs err: {abs_err}"

    # Test case 3: Complex function
    def complex_func(x):
        return ((x ** 2 + x) * Tensor.exp(x)).sum()

    x = Tensor(np.array([0.5, 1.0, 1.5]), requires_grad=True)
    match, rel_err, abs_err = check_gradients(complex_func, x)
    print(f"Complex function gradient check: Match={match}, Rel err={rel_err}, Abs err={abs_err}")
    assert match, f"Complex function gradient check failed. Rel err: {rel_err}, Abs err: {abs_err}"

if __name__ == "__main__":
    test_gradient_checker()