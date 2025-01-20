from typing import List, Optional, Union, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from graphviz import Digraph
import warnings

# Use type hints with Any to avoid circular imports
Tensor = Any

class GradientVisualizer:
    """Visualize gradients of tensors during training."""
    
    def __init__(self):
        self.gradient_history: Dict[str, List[float]] = {}
        self.iterations: List[int] = []
        self._iteration_count = 0
        
    def update(self, gradients: Dict[str, "Tensor"]) -> None:
        """
        Update gradient history with new values.
        
        Args:
            gradients: Dictionary mapping parameter names to their gradients
        """
        self._iteration_count += 1
        self.iterations.append(self._iteration_count)
        
        for name, grad in gradients.items():
            if grad is not None and hasattr(grad, 'grad') and grad.grad is not None:
                norm = float(np.linalg.norm(grad.grad.data))
                if name not in self.gradient_history:
                    self.gradient_history[name] = []
                self.gradient_history[name].append(norm)
    
    def plot(self, figsize: tuple = (10, 6), 
            log_scale: bool = True, 
            window_size: Optional[int] = None) -> None:
        """
        Plot gradient norms over iterations.
        
        Args:
            figsize: Figure size (width, height)
            log_scale: Whether to use log scale for y-axis
            window_size: Moving average window size for smoothing
        """
        plt.figure(figsize=figsize)
        
        for name, grad_history in self.gradient_history.items():
            if window_size is not None:
                # Apply moving average smoothing
                kernel = np.ones(window_size) / window_size
                smoothed = np.convolve(grad_history, kernel, mode='valid')
                plt.plot(self.iterations[window_size-1:], smoothed, label=name)
            else:
                plt.plot(self.iterations, grad_history, label=name)
        
        plt.xlabel('Iteration')
        plt.ylabel('Gradient Norm')
        if log_scale:
            plt.yscale('log')
        plt.title('Gradient Norms Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

class LossVisualizer:
    """Visualize training and validation loss curves."""
    
    def __init__(self):
        self.train_loss: List[float] = []
        self.val_loss: List[float] = []
        self.epochs: List[int] = []
        self._epoch_count = 0
        
    def update(self, train_loss: float, val_loss: Optional[float] = None) -> None:
        """
        Update loss history with new values.
        
        Args:
            train_loss: Training loss value
            val_loss: Validation loss value (optional)
        """
        self._epoch_count += 1
        self.epochs.append(self._epoch_count)
        self.train_loss.append(float(train_loss))
        if val_loss is not None:
            self.val_loss.append(float(val_loss))
    
    def plot(self, figsize: tuple = (10, 6), 
            log_scale: bool = False,
            window_size: Optional[int] = None) -> None:
        """
        Plot loss curves.
        
        Args:
            figsize: Figure size (width, height)
            log_scale: Whether to use log scale for y-axis
            window_size: Moving average window size for smoothing
        """
        plt.figure(figsize=figsize)
        
        def smooth(values: List[float]) -> np.ndarray:
            if window_size is not None:
                kernel = np.ones(window_size) / window_size
                return np.convolve(values, kernel, mode='valid')
            return np.array(values)
        
        x_train = self.epochs[window_size-1:] if window_size else self.epochs
        plt.plot(x_train, smooth(self.train_loss), label='Training Loss')
        
        if self.val_loss:
            x_val = self.epochs[window_size-1:] if window_size else self.epochs
            plt.plot(x_val, smooth(self.val_loss), label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        if log_scale:
            plt.yscale('log')
        plt.title('Training and Validation Loss Over Time')
        plt.legend()
        plt.grid(True)
        plt.show()

class ModelVisualizer:
    """Visualize neural network architecture using graphviz."""
    
    def __init__(self):
        self.dot = Digraph(comment='Neural Network')
        self.dot.attr(rankdir='LR')  # Left to right layout
        self._node_count = 0
        
    def _create_node_name(self) -> str:
        """Generate unique node identifier."""
        self._node_count += 1
        return f'node_{self._node_count}'
    
    def add_layer(self, name: str, 
                 input_shape: Union[tuple, List[int]], 
                 output_shape: Union[tuple, List[int]]) -> None:
        """
        Add a layer to the visualization.
        
        Args:
            name: Layer name/type
            input_shape: Shape of input tensor
            output_shape: Shape of output tensor
        """
        node_name = self._create_node_name()
        label = f'{name}\n{input_shape} → {output_shape}'
        self.dot.node(node_name, label, shape='box')
        
        # Connect to previous layer if exists
        if self._node_count > 1:
            prev_node = f'node_{self._node_count-1}'
            self.dot.edge(prev_node, node_name)
    
    def render(self, filename: str = 'model', 
              format: str = 'png',
              cleanup: bool = True) -> None:
        """
        Render the model visualization.
        
        Args:
            filename: Output filename (without extension)
            format: Output format ('png', 'pdf', etc.)
            cleanup: Whether to remove the source dot file
        """
        try:
            self.dot.render(filename, format=format, cleanup=cleanup)
        except Exception as e:
            warnings.warn(f"Failed to render model visualization: {str(e)}")

def plot_tensor(tensor: "Tensor", 
                title: Optional[str] = None,
                cmap: str = 'viridis',
                figsize: tuple = (8, 6)) -> None:
    """
    Visualize a 2D tensor as a heatmap.
    
    Args:
        tensor: 2D tensor to visualize
        title: Plot title
        cmap: Colormap to use
        figsize: Figure size (width, height)
    """
    if not hasattr(tensor, 'shape') or len(tensor.shape) != 2:
        raise ValueError("Can only visualize 2D tensors")
        
    plt.figure(figsize=figsize)
    plt.imshow(tensor.data, cmap=cmap)
    plt.colorbar()
    if title:
        plt.title(title)
    plt.show()

def plot_gradient_flow(named_parameters: Dict[str, "Tensor"]) -> None:
    """
    Plots the gradients flowing through different layers in the network during training.
    Can be used for checking for possible gradient vanishing / exploding problems.
    
    Args:
        named_parameters: Dictionary of parameter name to parameter tensor mapping
    """
    ave_grads = []
    max_grads = []
    layers = []
    
    for name, p in named_parameters.items():
        if hasattr(p, 'requires_grad') and p.requires_grad and hasattr(p, 'grad') and p.grad is not None:
            layers.append(name)
            ave_grads.append(np.mean(np.abs(p.grad.data)))
            max_grads.append(np.max(np.abs(p.grad.data)))
    
    if not layers:  # No gradients to plot
        warnings.warn("No gradients available to plot")
        return
        
    plt.figure(figsize=(10, 8))
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=45)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)
    plt.xlabel("Layers")
    plt.ylabel("Average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([
        plt.Rectangle((0,0),1,1,fc="c", alpha=0.1),
        plt.Rectangle((0,0),1,1,fc="b", alpha=0.1)
    ], ['max-gradient', 'mean-gradient'])
    plt.tight_layout()
    plt.show()