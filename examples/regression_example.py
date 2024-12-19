import numpy as np
from fiztorch.tensor import Tensor
from fiztorch.nn import Sequential, Linear
from fiztorch.optim import SGD
from fiztorch.utils import DataLoader

# Dummy regression data
X_train = np.random.rand(100, 1)
y_train = 2 * X_train + 1 + np.random.randn(100, 1) * 0.1

# DataLoader
train_loader = DataLoader(X_train, y_train, batch_size=32)

# Model
model = Sequential(
    Linear(1, 1)
)

# Optimizer
optimizer = SGD(model.parameters(), lr=0.01)

# Training loop
for epoch in range(10):
    for X_batch, y_batch in train_loader:
        X_batch = Tensor(X_batch)
        y_batch = Tensor(y_batch)

        optimizer.zero_grad()
        output = model(X_batch)
        loss = output.mse_loss(y_batch)
        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.data.item()}')
