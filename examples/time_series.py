"""
Time Series Prediction Example using FizTorch
Predicts future values of a sine wave with added noise
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))


import numpy as np
from fiztorch import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F
from fiztorch.optim import SGD
from fiztorch.utils import DataLoader

import matplotlib.pyplot as plt
from typing import Tuple, List

class TimeSeriesDataset:
    def __init__(self, seq_length: int = 50, num_samples: int = 1000, noise_level: float = 0.1):
        """
        Generate synthetic time series data based on sine waves with noise
        
        Args:
            seq_length: Length of each sequence
            num_samples: Number of sequences to generate
            noise_level: Amount of noise to add to the sine wave
        """
        self.seq_length = seq_length
        t = np.linspace(0, 8*np.pi, num_samples + seq_length)
        
        # Generate sine wave with multiple frequencies
        self.data = (np.sin(t) + 0.5 * np.sin(2*t) + 0.25 * np.sin(3*t))
        
        # Add noise
        self.data += np.random.normal(0, noise_level, self.data.shape)
        
        # Normalize data
        self.data = (self.data - self.data.mean()) / self.data.std()
        
    def get_sequence(self) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training"""
        X, y = [], []
        
        for i in range(len(self.data) - self.seq_length):
            X.append(self.data[i:(i + self.seq_length)])
            y.append(self.data[i + self.seq_length])
            
        return np.array(X), np.array(y)

class TimeSeriesPredictor(Sequential):
    def __init__(self, seq_length: int, hidden_size: int = 64):
        """
        Neural network for time series prediction
        
        Args:
            seq_length: Length of input sequence
            hidden_size: Number of hidden units
        """
        super().__init__(
            Linear(seq_length, hidden_size),
            ReLU(),
            Linear(hidden_size, hidden_size // 2),
            ReLU(),
            Linear(hidden_size // 2, 1)
        )

def train_predictor(model: TimeSeriesPredictor,
                   train_loader: DataLoader,
                   optimizer: SGD,
                   epochs: int = 50) -> List[float]:
    """
    Train the time series predictor
    
    Returns:
        List of training losses
    """
    losses = []
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        batch_count = 0
        
        for X_batch, y_batch in train_loader:
            # Convert to tensors
            X = Tensor(X_batch, requires_grad=True)
            y = Tensor(y_batch.reshape(-1, 1), requires_grad=True)
            
            # Forward pass
            pred = model(X)
            loss = F.mse_loss(pred, y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.data
            batch_count += 1
        
        avg_loss = epoch_loss / batch_count
        losses.append(avg_loss)
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {avg_loss:.6f}")
    
    return losses

def visualize_predictions(model: TimeSeriesPredictor,
                        dataset: TimeSeriesDataset,
                        num_predictions: int = 100):
    """Visualize the model's predictions against actual values"""
    X, y = dataset.get_sequence()
    
    # Get predictions for a subset of data
    test_X = X[-num_predictions:]
    test_y = y[-num_predictions:]
    
    # Make predictions
    predictions = []
    for seq in test_X:
        pred = model(Tensor(seq.reshape(1, -1))).data
        predictions.append(pred.item())
    
    # Plot results
    plt.figure(figsize=(12, 6))
    plt.plot(test_y, label='Actual', alpha=0.7)
    plt.plot(predictions, label='Predicted', alpha=0.7)
    plt.legend()
    plt.title('Time Series Prediction')
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.grid(True)
    plt.savefig('time_series_prediction.png')
    plt.close()

def main():
    # Parameters
    seq_length = 50
    batch_size = 32
    hidden_size = 64
    epochs = 500
    
    # Create dataset
    print("Generating dataset...")
    dataset = TimeSeriesDataset(seq_length=seq_length)
    X, y = dataset.get_sequence()
    
    # Create data loader
    data_loader = DataLoader(X, y, batch_size=batch_size, shuffle=True)
    
    # Initialize model and optimizer
    print("Initializing model...")
    model = TimeSeriesPredictor(seq_length, hidden_size)
    optimizer = SGD(model.parameters(), lr=0.01)
    
    # Train model
    print("Training model...")
    losses = train_predictor(model, data_loader, optimizer, epochs)
    
    # Visualize results
    print("Generating visualization...")
    visualize_predictions(model, dataset)
    print("Visualization saved as 'time_series_prediction.png'")
    
    # Plot training loss
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title('Training Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.grid(True)
    plt.savefig('training_loss.png')
    plt.close()
    print("Training loss plot saved as 'training_loss.png'")

if __name__ == "__main__":
    main()