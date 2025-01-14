import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from fiztorch import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
from fiztorch.optim.optimizer import SGD
import fiztorch.nn.functional as F

def generate_synthetic_data(n_samples=1000, noise_level=0.1):
    """
    Generates synthetic data with multiple patterns and controlled noise.
    
    Args:
        n_samples (int): Number of samples to generate
        noise_level (float): Amount of random noise to add
        
    Returns:
        tuple: (features, targets) as numpy arrays
    """
    # Generate input features
    x1 = np.random.uniform(-2, 2, n_samples)
    x2 = np.random.uniform(-2, 2, n_samples)
    
    # Create non-linear patterns
    # Pattern 1: Circular pattern
    radius = np.sqrt(x1**2 + x2**2)
    circle_component = np.sin(2 * radius)
    
    # Pattern 2: Interaction term
    interaction = 0.5 * x1 * x2
    
    # Pattern 3: Polynomial term
    polynomial = 0.3 * (x1**2 - x2**2)
    
    # Combine patterns and add noise
    y = (circle_component + interaction + polynomial +
         noise_level * np.random.randn(n_samples))
    
    # Normalize target values to [0, 1] range
    y = (y - y.min()) / (y.max() - y.min())
    
    # Reshape arrays
    X = np.column_stack((x1, x2))
    y = y.reshape(-1, 1)
    
    return X, y

def split_data(X, y, train_ratio=0.8):
    """
    Splits data into training and validation sets.
    
    Args:
        X (np.ndarray): Input features
        y (np.ndarray): Target values
        train_ratio (float): Ratio of data to use for training
        
    Returns:
        tuple: (X_train, y_train, X_val, y_val)
    """
    n_samples = len(X)
    indices = np.random.permutation(n_samples)
    split_idx = int(n_samples * train_ratio)
    
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]
    
    return (X[train_indices], y[train_indices],
            X[val_indices], y[val_indices])

model = Sequential(
    Linear(2, 64),
    ReLU(),
    Linear(64, 128),
    ReLU(),
    Linear(128, 64),
    ReLU(),
    Linear(64, 32),
    ReLU(),
    Linear(32, 1)
)

def evaluate_model(model, X, y):
    """
    Evaluates model performance on given data.
    
    Args:
        model (Sequential): The neural network model
        X (Tensor): Input features
        y (Tensor): Target values
        
    Returns:
        float: Mean squared error loss
    """
    predictions = model(X)
    loss = F.mse_loss(predictions, y)
    return loss.data

def log_training_progress(epoch, train_loss, val_loss=None):
    """
    Logs training progress with both training and validation metrics.
    
    Args:
        epoch (int): Current epoch number
        train_loss (float): Training loss
        val_loss (float, optional): Validation loss
    """
    print(f"\n=== Epoch {epoch + 1} ===")
    print(f"Training Loss: {train_loss:.6f}")
    if val_loss is not None:
        print(f"Validation Loss: {val_loss:.6f}")

def train_example(epochs=1000, batch_size=32, learning_rate=0.01):
    """
    Trains the model using mini-batch gradient descent.
    
    Args:
        epochs (int): Number of training epochs
        batch_size (int): Size of mini-batches
        learning_rate (float): Learning rate for optimization
    """
    # Generate and split synthetic data
    X, y = generate_synthetic_data(n_samples=1000)
    X_train, y_train, X_val, y_val = split_data(X, y)
    
    # Convert to Tensors
    X_train = Tensor(X_train, requires_grad=True)
    y_train = Tensor(y_train)
    X_val = Tensor(X_val, requires_grad=True)
    y_val = Tensor(y_val)
    
    # Initialize optimizer
    optimizer = SGD(model.parameters(), lr=learning_rate)
    
    n_samples = len(X_train)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    for epoch in range(epochs):
        total_train_loss = 0
        
        # Mini-batch training
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, n_samples)
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            
            optimizer.zero_grad()
            predictions = model(batch_X)
            loss = F.mse_loss(predictions, batch_y)
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.data
        
        # Calculate average training loss
        avg_train_loss = total_train_loss / n_batches
        
        # Evaluate on validation set
        val_loss = evaluate_model(model, X_val, y_val)
        
        # Log progress
        log_training_progress(epoch, avg_train_loss, val_loss)

if __name__ == "__main__":
    train_example()