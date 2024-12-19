import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.nn import Linear, ReLU, Module
from fiztorch.optim import SGD
from fiztorch.utils.data import DataLoader

class MNISTClassifier(Module):
    def __init__(self):
        super().__init__()
        self.fc1 = Linear(784, 128)
        self.relu1 = ReLU()
        self.fc2 = Linear(128, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

def main():
    # Generate dummy MNIST-like data for example
    num_samples = 1000
    X = np.random.randn(num_samples, 784)
    y = np.random.randint(0, 10, size=num_samples)

    # Create data loader
    train_loader = DataLoader(X, y, batch_size=32)

    # Initialize model and optimizer
    model = MNISTClassifier()
    optimizer = SGD(model.parameters(), lr=0.01)

    # Training loop
    num_epochs = 5
    for epoch in range(num_epochs):
        total_loss = 0
        for batch_data, batch_labels in train_loader:
            # Convert to tensors
            inputs = Tensor(batch_data)
            labels = Tensor(batch_labels)

            # Forward pass
            outputs = model(inputs)
            
            # Compute loss (simple MSE for example)
            loss = ((outputs - labels) * (outputs - labels)).sum()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update weights
            optimizer.step()
            
            total_loss += loss.data

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

if __name__ == "__main__":
    main()