import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from fiztorch.tensor import Tensor
from fiztorch.nn.layers import Linear, ReLU
from fiztorch.nn.sequential import Sequential
import fiztorch.nn.functional as F
import fiztorch.optim.optimizer as opt
from fiztorch.utils import visual

def load_wine_data():
    try:
        # Load the wine dataset
        wine = load_wine()
        X, y = wine.data, wine.target

        # Normalize the data
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Verify data shapes before converting to tensors
        n_features = X_train.shape[1]
        n_classes = len(np.unique(y))
        assert n_features == 13, f"Expected 13 features, got {n_features}"
        assert n_classes == 3, f"Expected 3 classes, got {n_classes}"

        return (Tensor(X_train), Tensor(y_train), 
                Tensor(X_test), Tensor(y_test))
    except Exception as e:
        print(f"Error loading Wine data: {str(e)}")
        raise

def create_model():
    try:
        model = Sequential(
            Linear(13, 64),    # 13 input features for wine dataset
            ReLU(),
            Linear(64, 32),
            ReLU(),
            Linear(32, 3),     # 3 classes for wine quality
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

def main():
    try:
        # Load data
        print("Loading Wine Quality data...")
        X_train, y_train, X_test, y_test = load_wine_data()

        # Create model and optimizer
        print("Creating model...")
        model = create_model()
        optimizer = opt.Adam(model.parameters(), lr=0.001)

        # Training parameters
        n_epochs = 200  # Reduced epochs as wine dataset is smaller
        batch_size = 16  # Smaller batch size for better generalization

        train_losses = []
        train_accuracies = []
        test_accuracies = []
        loss_visual = visual.LossVisualizer()

        # Training loop
        print("Training started...")
        for epoch in range(n_epochs):
            avg_loss = train_epoch(model, optimizer, X_train, y_train, batch_size, epoch)
            train_acc = evaluate(model, X_train, y_train)
            test_acc = evaluate(model, X_test, y_test)

            loss_visual.update(avg_loss)
            train_losses.append(avg_loss)
            train_accuracies.append(train_acc)
            test_accuracies.append(test_acc)

            if (epoch + 1) % 10 == 0:  # Print every 10 epochs
                print(f"Epoch {epoch + 1}")
                print(f"Average Loss: {avg_loss:.4f}")
                print(f"Training Accuracy: {train_acc:.4f}")
                print(f"Test Accuracy: {test_acc:.4f}")
                print("-" * 50)

        print("Training complete!")
        loss_visual.plot(window_size=5)

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