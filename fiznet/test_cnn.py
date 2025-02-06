import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

from cnn import CNN

def load_mnist(num_samples=None):
    """Load MNIST dataset and preprocess it."""
    print("Loading MNIST dataset...")
    X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
    
    if num_samples is not None:
        X = X[:num_samples]
        y = y[:num_samples]
    
    # Convert to integers
    y = y.astype(np.int32)
    
    # Reshape and scale the data
    X = X.reshape(-1, 28, 28, 1) / 255.0
    
    return X, y

def batch_iterator(X, y, batch_size):
    """Create batches of data."""
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    
    for i in range(0, len(X), batch_size):
        batch_indices = indices[i:i + batch_size]
        yield X[batch_indices], y[batch_indices]

def evaluate(model, X, y, batch_size=32):
    """Evaluate the model on given data."""
    total_loss = 0
    correct = 0
    batches = 0
    
    for X_batch, y_batch in batch_iterator(X, y, batch_size):
        # Forward pass
        logits = model.forward(X_batch)
        loss = model.loss.forward(logits, y_batch)
        
        # Calculate accuracy
        predictions = np.argmax(logits, axis=1)
        correct += np.sum(predictions == y_batch)
        
        total_loss += loss
        batches += 1
    
    avg_loss = total_loss / batches
    accuracy = correct / len(X)
    
    return avg_loss, accuracy

def train_model(model, X_train, y_train, X_val, y_val, epochs=10, batch_size=32, learning_rate=0.001):
    """Train the CNN model."""
    train_losses = []
    val_losses = []
    train_accuracies = []
    val_accuracies = []
    
    for epoch in range(epochs):
        epoch_start = time.time()
        total_loss = 0
        batches = 0
        
        # Training loop
        for X_batch, y_batch in batch_iterator(X_train, y_train, batch_size):
            # Forward pass
            loss = model.compute_loss(X_batch, y_batch)
            
            # Backward pass
            model.backward_loss()
            
            # Update parameters
            model.update_params(lr=learning_rate)
            
            total_loss += loss
            batches += 1
            
            # Print progress every 100 batches
            if batches % 100 == 0:
                print(f"Epoch {epoch + 1}, Batch {batches}, Loss: {loss:.4f}")
        
        # Calculate training metrics
        train_loss, train_acc = evaluate(model, X_train, y_train, batch_size)
        val_loss, val_acc = evaluate(model, X_val, y_val, batch_size)
        
        # Store metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accuracies.append(train_acc)
        val_accuracies.append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print(f"Time: {epoch_time:.2f}s")
        print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}\n")
    
    return {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accuracies': train_accuracies,
        'val_accuracies': val_accuracies
    }

def main():
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Load and preprocess data
    X, y = load_mnist(num_samples=10000)  # Using 10k samples for faster testing
    
    # Split the data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Initialize the model
    model = CNN()
    
    # Training parameters
    epochs = 5
    batch_size = 32
    learning_rate = 0.001
    
    print("Starting training...")
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    print(f"Epochs: {epochs}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}\n")
    
    # Train the model
    history = train_model(
        model,
        X_train, y_train,
        X_val, y_val,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate
    )
    
    # Final evaluation
    train_loss, train_acc = evaluate(model, X_train, y_train)
    val_loss, val_acc = evaluate(model, X_val, y_val)
    
    print("\nFinal Results:")
    print(f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()