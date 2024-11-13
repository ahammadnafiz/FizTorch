import os
import sys
import numpy as np

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import modules (assuming `ftensor` is structured with nn, optim, data, and utils modules)
from ftensor import nn, optim, data, utils

# Define the SimpleDataset and DataLoader if not already defined in data module
class SimpleDataset(data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

class DataLoader:
    def __init__(self, dataset, batch_size=8, shuffle=False):
        self.bs = batch_size
        self.dataset = dataset
        self.shuffle = shuffle
        self.idx = np.arange(len(self.dataset))
        self.make_indexes()

    def make_indexes(self):
        self.index = 0
        if self.shuffle:
            np.random.shuffle(self.idx)

    def __len__(self):
        return (len(self.dataset) // self.bs)

    def __next__(self):
        if self.index < (len(self.dataset) // self.bs):
            slc = slice(self.index * self.bs, (self.index + 1) * self.bs)
            self.index += 1
            # Unpack the batch into separate arrays for X and y
            batch = [self.dataset[i] for i in self.idx[slc]]
            return np.array([item[0] for item in batch]), np.array([item[1] for item in batch])  # Unpacking
        else:
            self.make_indexes()
            raise StopIteration

    def __iter__(self):
        return self

# Create dataset and dataloader
X = np.random.randn(1000, 10)
y = np.random.randint(0, 2, (1000, 1))
dataset = SimpleDataset(list(zip(X, y)))
dataloader = DataLoader(dataset, batch_size=32)

# Create a model
model = nn.Sequential(
    nn.Linear(10, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, 1),
    nn.Sigmoid()
)

# Define loss function and optimizer
loss_fn = lambda pred, target: ((pred - target) ** 2).mean()
optimizer = optim.Adam(model.parameters(), learning_rate=0.001)

def train(model, dataloader, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        for batch in dataloader:
            # Unpack batch into separate input and target arrays
            X_batch, y_batch = zip(*batch)
            X_batch, y_batch = np.array(X_batch), np.array(y_batch)

            # Forward pass
            predictions = model.forward(X_batch)
            loss = loss_fn(predictions, y_batch)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}")

# Train the model
utils.train(model, dataloader, loss_fn, optimizer, epochs=10)
print("Training completed!")

