import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.nn import functional as F
from fiztorch.utils import visual

def test_gradient_visualizer():
    """Test the GradientVisualizer class."""
    print("Testing GradientVisualizer...")
    
    # Create a GradientVisualizer instance
    grad_viz = visual.GradientVisualizer()
    
    # Simulate training iterations with different gradients
    for i in range(100):
        # Create synthetic gradients
        gradients = {
            'layer1': Tensor(np.random.randn(10, 10) / (i + 1), requires_grad=True),
            'layer2': Tensor(np.random.randn(5, 5) / (i + 1), requires_grad=True)
        }
        
        # Add gradients to tensors
        for name, tensor in gradients.items():
            tensor.grad = Tensor(np.random.randn(*tensor.shape))
        
        # Update visualizer
        grad_viz.update(gradients)
    
    # Plot gradients with different options
    print("Plotting gradient norms...")
    grad_viz.plot(log_scale=True)
    grad_viz.plot(log_scale=True, window_size=5)  # With smoothing

def test_loss_visualizer():
    """Test the LossVisualizer class."""
    print("\nTesting LossVisualizer...")
    
    # Create a LossVisualizer instance
    loss_viz = visual.LossVisualizer()
    
    # Simulate training epochs
    for epoch in range(50):
        # Generate synthetic loss values
        train_loss = 1.0 / (epoch + 1) + 0.1 * np.random.rand()
        val_loss = 1.2 / (epoch + 1) + 0.1 * np.random.rand()
        
        # Update visualizer
        loss_viz.update(train_loss, val_loss)
    
    # Plot loss curves
    print("Plotting loss curves...")
    loss_viz.plot()
    loss_viz.plot(window_size=5)  # With smoothing

def test_model_visualizer():
    """Test the ModelVisualizer class."""
    print("\nTesting ModelVisualizer...")
    
    # Create a ModelVisualizer instance
    model_viz = visual.ModelVisualizer()
    
    # Add layers to visualize a simple neural network
    model_viz.add_layer("Input", (32, 784), (32, 784))
    model_viz.add_layer("Linear1", (32, 784), (32, 256))
    model_viz.add_layer("ReLU1", (32, 256), (32, 256))
    model_viz.add_layer("Linear2", (32, 256), (32, 10))
    model_viz.add_layer("Softmax", (32, 10), (32, 10))
    
    # Render the visualization
    print("Rendering model visualization...")
    model_viz.render("test_model")

def test_tensor_visualization():
    """Test tensor visualization utilities."""
    print("\nTesting tensor visualization...")
    
    # Create a 2D tensor with a pattern
    x = np.linspace(-5, 5, 100)
    y = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.sin(np.sqrt(X**2 + Y**2))
    tensor = Tensor(Z)
    
    # Visualize the tensor
    print("Plotting tensor heatmap...")
    visual.plot_tensor(tensor, title="Tensor Visualization", cmap="coolwarm")

def test_gradient_flow():
    """Test gradient flow visualization."""
    print("\nTesting gradient flow visualization...")
    
    # Create synthetic parameters with gradients
    parameters = {
        'conv1.weight': Tensor(np.random.randn(16, 3, 3, 3), requires_grad=True),
        'conv1.bias': Tensor(np.random.randn(16), requires_grad=True),
        'conv2.weight': Tensor(np.random.randn(32, 16, 3, 3), requires_grad=True),
        'conv2.bias': Tensor(np.random.randn(32), requires_grad=True),
        'fc1.weight': Tensor(np.random.randn(128, 512), requires_grad=True),
        'fc1.bias': Tensor(np.random.randn(128), requires_grad=True),
        'fc2.weight': Tensor(np.random.randn(10, 128), requires_grad=True),
        'fc2.bias': Tensor(np.random.randn(10), requires_grad=True)
    }
    
    # Add synthetic gradients
    for param in parameters.values():
        param.grad = Tensor(np.random.randn(*param.shape) * 0.01)
    
    # Visualize gradient flow
    print("Plotting gradient flow...")
    visual.plot_gradient_flow(parameters)

def main():
    """Run all visualization tests."""
    print("Starting visualization tests...\n")
    
    # Run all tests
    test_gradient_visualizer()
    test_loss_visualizer()
    test_model_visualizer()
    test_tensor_visualization()
    test_gradient_flow()
    
    print("\nAll visualization tests completed!")

if __name__ == "__main__":
    main()