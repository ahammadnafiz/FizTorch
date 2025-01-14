import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from matplotlib.animation import FuncAnimation, PillowWriter
import matplotlib.gridspec as gridspec

from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Sequential
import fiztorch.nn.functional as F
import fiztorch.optim.optimizer as opt

# Original mnist digit data from openml
# def load_mnist_data():
#     try:
#         # Fetch the MNIST dataset from OpenML
#         mnist = fetch_openml('mnist_784', version=1)
#         X, y = mnist.data, mnist.target.astype(int)

#         # Normalize the data
#         scaler = StandardScaler()
#         X = scaler.fit_transform(X)

#         # Split the data into training and testing sets
#         X_train, X_test, y_train, y_test = train_test_split(
#             X, y, test_size=0.2, random_state=42
#         )

#         # Verify data shapes before converting to tensors
#         assert X_train.shape[1] == 784, "Input features should be 784-dimensional"
#         assert len(np.unique(y)) == 10, "Should have 10 classes"

#         return (Tensor(X_train), Tensor(y_train), 
#                 Tensor(X_test), Tensor(y_test))
#     except Exception as e:
#         print(f"Error loading MNIST data: {str(e)}")
#         raise

# def create_model():
#     try:
#         model = Sequential(
#             Linear(784, 128),  # Updated input size to 784
#             ReLU(),
#             Linear(128, 64),
#             ReLU(),
#             Linear(64, 10)
#         )
#         return model
#     except Exception as e:
#         print(f"Error creating model: {str(e)}")
#         raise


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
            Linear(64, 10),
        )
        return model
    except Exception as e:
        print(f"Error creating model: {str(e)}")
        raise

def print_parameter_gradients(model, epoch):
    """
    Print gradients for all parameters in the model
    """
    print(f"\nEpoch {epoch + 1} Parameter Gradients:")
    for idx, layer in enumerate(model.layers):
        if hasattr(layer, '_parameters'):
            for param_name, param in layer._parameters.items():
                if param.grad is not None:
                    grad_stats = {
                        'mean': float(param.grad.data.mean()),
                        'std': float(param.grad.data.std()),
                        'max': float(param.grad.data.max()),
                        'min': float(param.grad.data.min())
                    }
                    print(f"Layer {idx} - {param_name}:")
                    print(f"  Gradient stats: {grad_stats}")

def train_epoch(model, optimizer, X_train, y_train, batch_size, epoch):
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
            
            # Print gradients for the last batch of each epoch
            if i + batch_size >= len(X_train.data):
                print_parameter_gradients(model, epoch)
                
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

def create_training_animation(train_losses, train_accuracies, test_accuracies, save_path='training_progress.gif'):
    # Create figure and subplots
    fig = plt.figure(figsize=(12, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    # Maximum values for scaling
    max_loss = max(train_losses)
    max_acc = max(max(train_accuracies), max(test_accuracies))

    def animate(frame):
        ax1.clear()
        ax2.clear()

        # Get data up to current frame
        current_losses = train_losses[:frame+1]
        current_train_acc = train_accuracies[:frame+1]
        current_test_acc = test_accuracies[:frame+1]
        epochs = range(1, frame+2)

        # Plot Loss
        ax1.plot(epochs, current_losses, 'b-', label='Train Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss over Epochs')
        ax1.legend()
        ax1.set_ylim(0, max_loss * 1.1)
        ax1.grid(True, linestyle='--', alpha=0.7)

        # Plot Accuracy
        ax2.plot(epochs, current_train_acc, 'g-', label='Train Accuracy')
        ax2.plot(epochs, current_test_acc, 'r-', label='Test Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.set_title('Accuracy over Epochs')
        ax2.legend()
        ax2.set_ylim(0, max_acc * 1.1)
        ax2.grid(True, linestyle='--', alpha=0.7)

        plt.tight_layout()

    # Create animation
    n_frames = len(train_losses)
    anim = FuncAnimation(
        fig, 
        animate, 
        frames=n_frames,
        interval=50,  # 50ms between frames
        repeat=True
    )

    # Save as GIF
    writer = PillowWriter(fps=20)
    anim.save(save_path, writer=writer)
    plt.close()

def main():
    try:
        # Load data
        print("Loading MNIST data...")
        X_train, y_train, X_test, y_test = load_mnist_data()

        # Create model and optimizer
        print("Creating model...")
        model = create_model()
        optimizer = opt.Adam(model.parameters(), lr=0.001)

        # Training parameters
        n_epochs = 2000  # Reduced for clarity of output
        batch_size = 32

        train_losses = []
        train_accuracies = []
        test_accuracies = []

        # Training loop
        print("Training started...")
        for epoch in range(n_epochs):
            avg_loss = train_epoch(model, optimizer, X_train, y_train, batch_size, epoch)
            train_acc = evaluate(model, X_train, y_train)
            test_acc = evaluate(model, X_test, y_test)

            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            print(f"Epoch {epoch + 1}")
            print(f"Average Loss: {avg_loss:.4f}")
            print(f"Training Accuracy: {train_acc:.4f}")
            print(f"Test Accuracy: {test_acc:.4f}")
            print("-" * 50)

        # create_training_animation(train_losses, train_accuracies, test_accuracies)

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
