import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential, Module
from fiztorch.optim import SGD
from fiztorch.utils.data import DataLoader
import fiztorch.nn.functional as F

class DebugLinear(Linear):
    def forward(self, x):
        # Ensure weights have requires_grad=True
        if self.weight.requires_grad is False:
            self.weight = Tensor(self.weight.data, requires_grad=True)
        if self.bias.requires_grad is False:
            self.bias = Tensor(self.bias.data, requires_grad=True)
            
        # Compute output with gradient tracking
        output = x @ self.weight.T + self.bias
        
        print(f"Linear layer forward:")
        print(f"- Input shape: {x.shape}")
        print(f"- Weight shape: {self.weight.shape}")
        print(f"- Output shape: {output.shape}")
        print(f"- Weight requires grad: {self.weight.requires_grad}")
        print(f"- Weight grad exists: {self.weight.grad is not None}")
        if self.weight.grad is not None:
            print(f"- Weight grad mean: {np.mean(self.weight.grad.data)}")
            print(f"- Weight grad std: {np.std(self.weight.grad.data)}")
        return output

def debug_tensor(tensor, name):
    print(f"\nDebug {name}:")
    print(f"- Shape: {tensor.shape}")
    print(f"- Mean: {np.mean(tensor.data)}")
    print(f"- Std: {np.std(tensor.data)}")
    print(f"- Min: {np.min(tensor.data)}")
    print(f"- Max: {np.max(tensor.data)}")
    print(f"- Requires grad: {tensor.requires_grad}")
    print(f"- Has grad: {tensor.grad is not None}")
    if tensor.grad is not None:
        print(f"- Grad mean: {np.mean(tensor.grad.data)}")
        print(f"- Grad std: {np.std(tensor.grad.data)}")

class CustomCrossEntropy:
    def __call__(self, logits: Tensor, targets: Tensor) -> Tensor:
        batch_size = logits.shape[0]
        # Compute softmax
        max_logits = np.max(logits.data, axis=1, keepdims=True)
        exp_logits = np.exp(logits.data - max_logits)
        probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Compute cross entropy loss
        target_probs = probs[np.arange(batch_size), targets.data.astype(int)]
        loss = -np.mean(np.log(target_probs + 1e-8))
        
        # Create output tensor with gradient function
        out = Tensor(loss, requires_grad=logits.requires_grad)
        
        if logits.requires_grad:
            def _backward(grad_output):
                # Compute gradient of cross entropy w.r.t. logits
                grad = probs.copy()
                grad[np.arange(batch_size), targets.data.astype(int)] -= 1
                grad = grad / batch_size
                logits.backward(Tensor(grad * grad_output.data))
            
            out._grad_fn = _backward
            out.is_leaf = False
            
        return out

class MNISTClassifier(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = DebugLinear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = DebugLinear(128, 64)
        self.relu2 = ReLU()
        self.fc3 = DebugLinear(64, 10)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x
    
    def parameters(self):
        return [self.fc1.weight, self.fc1.bias,
                self.fc2.weight, self.fc2.bias,
                self.fc3.weight, self.fc3.bias]

def train_one_batch(model, optimizer, batch_data, batch_labels, criterion):
    """Train on a single batch with detailed debugging."""
    # Zero gradients
    optimizer.zero_grad()
    
    # Forward pass
    inputs = Tensor(batch_data, requires_grad=True)
    targets = Tensor(batch_labels.astype(np.int64), requires_grad=False)
    
    debug_tensor(inputs, "inputs")
    
    # Get logits
    logits = model(inputs)
    debug_tensor(logits, "logits")
    
    # Compute loss
    loss = criterion(logits, targets)
    print(f"\nLoss value: {loss.data}")
    
    # Backward pass
    loss.backward()
    
    # Debug gradients before update
    print("\nGradients before update:")
    for i, param in enumerate(model.parameters()):
        debug_tensor(param, f"parameter {i}")
    
    # Update weights
    optimizer.step()
    
    # Debug parameters after update
    print("\nParameters after update:")
    for i, param in enumerate(model.parameters()):
        debug_tensor(param, f"parameter {i}")
    
    return loss.data

def main():
    # Load a very small subset of MNIST for debugging
    print("Loading MNIST dataset...")
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
    X = mnist["data"].astype(np.float32)[:1000]  # Just take first 1000 samples
    y = mnist["target"].astype(np.int64)[:1000]
    
    # Normalize data
    X = X / 255.0
    
    # Create very small batch for debugging
    batch_size = 4
    train_loader = DataLoader(X, y, batch_size=batch_size, shuffle=True)
    
    # Initialize model, optimizer and criterion
    model = MNISTClassifier()
    optimizer = SGD(model.parameters(), lr=0.01)
    criterion = CustomCrossEntropy()
    
    # Train on just one batch for debugging
    print("\nTraining on one batch...")
    for batch_data, batch_labels in train_loader:
        loss = train_one_batch(model, optimizer, batch_data, batch_labels, criterion)
        print(f"\nFinal loss: {loss}")
        break

if __name__ == "__main__":
    main()