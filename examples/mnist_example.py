import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F
from fiztorch.optim.optimizer import SGD

def load_mnist_data():
    try:
        digits = load_digits()
        X, y = digits.data, digits.target
        
        # Normalize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Verify data shapes before converting to tensors
        assert X_train.shape[1] == 64, "Input features should be 64-dimensional"
        assert len(np.unique(y)) == 10, "Should have 10 classes"
        
        return (Tensor(X_train), Tensor(y_train), 
                Tensor(X_test), Tensor(y_test))
    except Exception as e:
        print(f"Error loading MNIST data: {str(e)}")
        raise

def create_model():
    try:
        model = Sequential(
            Linear(64, 128),
            ReLU(),
            Linear(128, 64),
            ReLU(),
            Linear(64, 10)
        )
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def train_epoch(model, optimizer, X_train, y_train, batch_size):
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
            logits = model(X_batch)
            loss = F.cross_entropy(logits, y_batch)
            
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
    try:
        logits = model(X)
        probs = F.softmax(logits, dim=-1)
        predictions = np.argmax(probs.data, axis=1)
        accuracy = np.mean(predictions == y.data)
        return accuracy
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        raise

def main():
    try:
        # Load data
        print("Loading MNIST data...")
        X_train, y_train, X_test, y_test = load_mnist_data()
        
        # Create model and optimizer
        print("Creating model...")
        model = create_model()
        optimizer = SGD(model.parameters(), lr=0.01)
        
        # Training parameters
        n_epochs = 500
        batch_size = 32
        
        # Training loop
        print("Training started...")
        for epoch in range(n_epochs):
            avg_loss = train_epoch(model, optimizer, X_train, y_train, batch_size)
            train_acc = evaluate(model, X_train, y_train)
            test_acc = evaluate(model, X_test, y_test)
            
            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch + 1}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Training Accuracy: {train_acc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}")
                print("-" * 50)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        raise