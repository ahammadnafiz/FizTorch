import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import numpy as np
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
from fiztorch.optim.optimizer import SGD

def load_housing_data():
    """Load and preprocess California Housing dataset"""
    try:
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        # Normalize features and target
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X = X_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y.reshape(-1, 1))
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        return (Tensor(X_train), Tensor(y_train), 
                Tensor(X_test), Tensor(y_test),
                X_scaler, y_scaler)
    except Exception as e:
        print(f"Error loading housing data: {str(e)}")
        raise

def create_model(input_dim):
    """Create a regression model"""
    try:
        model = Sequential(
            Linear(input_dim, 64),
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 1)
        )
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def mse_loss(predictions: Tensor, targets: Tensor) -> Tensor:
    """Calculate Mean Squared Error loss"""
    return ((predictions - targets) ** 2).mean()

def train_epoch(model, optimizer, X_train, y_train, batch_size):
    """Train for one epoch"""
    try:
        indices = np.random.permutation(len(X_train.data))
        total_loss = 0
        n_batches = 0
        
        for i in range(0, len(X_train.data), batch_size):
            batch_indices = indices[i:i+batch_size]
            X_batch = Tensor(X_train.data[batch_indices])
            y_batch = Tensor(y_train.data[batch_indices])
            
            optimizer.zero_grad()
            
            # Forward pass
            predictions = model(X_batch)
            loss = mse_loss(predictions, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.data
            n_batches += 1
        
        return total_loss / n_batches
    except Exception as e:
        print(f"Error during training epoch: {str(e)}")
        raise

def evaluate(model, X, y):
    """Calculate MSE on the dataset"""
    try:
        predictions = model(X)
        mse = mse_loss(predictions, y)
        return mse.data
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

# Tests
def test_data_loading():
    X_train, y_train, X_test, y_test, _, _ = load_housing_data()
    assert len(X_train.shape) == 2, "Features should be 2D"
    assert len(y_train.shape) == 2, "Targets should be 2D"
    assert y_train.shape[1] == 1, "Target should have 1 dimension"

def test_model_creation():
    input_dim = 8  # California housing has 8 features
    model = create_model(input_dim)
    assert isinstance(model, Sequential), "Model should be Sequential"
    assert len(model.layers) == 5, "Model should have 5 layers"

def test_forward_pass():
    input_dim = 8
    model = create_model(input_dim)
    X_train, _, _, _, _, _ = load_housing_data()
    output = model(X_train)
    assert output.shape[1] == 1, "Output should have 1 dimension"

def test_mse_loss():
    pred = Tensor([[1.0], [2.0], [3.0]])
    target = Tensor([[1.1], [2.1], [2.9]])
    loss = mse_loss(pred, target)
    assert isinstance(loss.data, float) or isinstance(loss.data, np.ndarray)
    assert loss.data > 0, "MSE loss should be positive"

def test_training_step():
    input_dim = 8
    model = create_model(input_dim)
    X_train, y_train, _, _, _, _ = load_housing_data()
    optimizer = SGD(model.parameters(), lr=0.01)
    
    loss = train_epoch(model, optimizer, X_train, y_train, batch_size=32)
    assert isinstance(loss, float) or isinstance(loss, np.ndarray)
    assert not np.isnan(loss), "Loss should not be NaN"

def main():
    try:
        # Load data
        print("Loading California Housing dataset...")
        X_train, y_train, X_test, y_test, _, _ = load_housing_data()
        
        # Create model and optimizer
        print("Creating model...")
        input_dim = X_train.shape[1]
        model = create_model(input_dim)
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # Training parameters
        n_epochs = 500
        batch_size = 32
        
        # Training loop
        print("Training started...")
        best_test_mse = float('inf')
        
        for epoch in range(n_epochs):
            avg_loss = train_epoch(model, optimizer, X_train, y_train, batch_size)
            train_mse = evaluate(model, X_train, y_train)
            test_mse = evaluate(model, X_test, y_test)
            
            if test_mse < best_test_mse:
                best_test_mse = test_mse
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Training MSE: {train_mse:.4f}")
                print(f"Test MSE: {test_mse:.4f}")
                print(f"Best Test MSE: {best_test_mse:.4f}")
                print("-" * 50)
                
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        pytest.main([__file__])
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise