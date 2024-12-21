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
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

# Store metrics for animation
train_mse_history = []
test_mse_history = []
epochs_history = []

def load_housing_data():
    """Load and preprocess California Housing dataset"""
    try:
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        X_scaler = StandardScaler()
        y_scaler = StandardScaler()
        
        X = X_scaler.fit_transform(X)
        y = y_scaler.fit_transform(y.reshape(-1, 1))
        
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
            predictions = model(X_batch)
            loss = mse_loss(predictions, y_batch)
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

def create_training_animation(save_path='training_progress.gif'):
    """Create and save animation of training progress"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Mean Squared Error')
        ax.set_title('Training Progress')
        
        # Plot data up to current frame
        ax.plot(epochs_history[:frame], train_mse_history[:frame], 
                label='Training MSE', color='blue')
        ax.plot(epochs_history[:frame], test_mse_history[:frame], 
                label='Test MSE', color='red')
        
        ax.legend()
        ax.grid(True)
        
        # Set y-axis limits based on data range
        if len(train_mse_history) > 0:
            max_mse = max(max(train_mse_history), max(test_mse_history))
            min_mse = min(min(train_mse_history), min(test_mse_history))
            ax.set_ylim(min_mse * 0.9, max_mse * 1.1)
    
    # Create animation
    anim = FuncAnimation(
        fig, animate, 
        frames=len(epochs_history), 
        interval=50,  # 50ms between frames
        repeat=False
    )
    
    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(save_path, writer=writer)
    plt.close()

def main():
    try:
        # Clear previous history
        train_mse_history.clear()
        test_mse_history.clear()
        epochs_history.clear()
        
        # Load data
        print("Loading California Housing dataset...")
        X_train, y_train, X_test, y_test, _, _ = load_housing_data()
        
        # Create model and optimizer
        print("Creating model...")
        input_dim = X_train.shape[1]
        model = create_model(input_dim)
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # Training parameters
        n_epochs = 200
        batch_size = 32
        
        # Training loop
        print("Training started...")
        best_test_mse = float('inf')
        
        for epoch in range(n_epochs):
            avg_loss = train_epoch(model, optimizer, X_train, y_train, batch_size)
            train_mse = evaluate(model, X_train, y_train)
            test_mse = evaluate(model, X_test, y_test)
            
            # Store metrics for animation
            train_mse_history.append(train_mse)
            test_mse_history.append(test_mse)
            epochs_history.append(epoch)
            
            if test_mse < best_test_mse:
                best_test_mse = test_mse
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch + 1}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Training MSE: {train_mse:.4f}")
                print(f"Test MSE: {test_mse:.4f}")
                print(f"Best Test MSE: {best_test_mse:.4f}")
                print("-" * 50)
        
        # Create and save animation
        print("Creating training animation...")
        create_training_animation()
        print("Animation saved as 'training_progress.gif'")
                
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