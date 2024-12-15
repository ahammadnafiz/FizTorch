import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ftensor.core.tensor import Tensor

# Set random seed for reproducibility
np.random.seed(42)

def test_simple_gradients():
    # Create small tensors for easy verification
    x = Tensor(np.array([[1.0]]), requires_grad=True)
    y = Tensor(np.array([[2.0]]), requires_grad=True)
    z = Tensor(np.array([[3.0]]), requires_grad=True)
    
    print("Initial values:")
    print("x:", x.data[0,0])
    print("y:", y.data[0,0])
    print("z:", z.data[0,0])
    
    # Forward pass
    a = x + y  # a = 3.0
    print("\nAfter a = x + y:")
    print("a:", a.data[0,0])
    
    b = a * z  # b = 9.0
    print("After b = a * z:")
    print("b:", b.data[0,0])
    
    c = b.relu()  # c = 9.0 (since b > 0)
    print("After c = b.relu():")
    print("c:", c.data[0,0])
    
    d = c.sum()  # d = 9.0
    print("After d = c.sum():")
    print("d:", d.data)
    
    # Backward pass
    d.backward()
    
    print("\nGradients:")
    print("x.grad:", x.grad.data[0,0] if x.grad is not None else None)  # Should be 3.0
    print("y.grad:", y.grad.data[0,0] if y.grad is not None else None)  # Should be 3.0
    print("z.grad:", z.grad.data[0,0] if z.grad is not None else None)  # Should be 3.0

def test_larger_gradients():
    # Test with the original dimensions
    np.random.seed(42)
    x = Tensor(np.random.randn(10, 5), requires_grad=True)
    y = Tensor(np.random.randn(10, 5), requires_grad=True)
    z = Tensor(np.random.randn(10, 5), requires_grad=True)
    
    # Forward pass
    a = x + y
    b = a * z
    c = b.relu()
    d = c.sum()
    
    # Backward pass
    d.backward()
    print(d.data)
    
    print("\nLarger tensor test:")
    print("x.grad exists:", x.grad is not None)
    print("y.grad exists:", y.grad is not None)
    print("z.grad exists:", z.grad is not None)
    print("Sample gradients at [0,0]:")
    print("x.grad[0,0]:", x.grad.data[1,0] if x.grad is not None else None)
    print("y.grad[0,0]:", y.grad.data[0,2] if y.grad is not None else None)
    print("z.grad[0,0]:", z.grad.data[0,0] if z.grad is not None else None)

if __name__ == "__main__":
    print("Testing with simple values:")
    print("-" * 40)
    test_simple_gradients()
    
    print("\nTesting with larger tensors:")
    print("-" * 40)
    test_larger_gradients()