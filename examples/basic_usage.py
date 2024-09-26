# Example usage (can be placed in a separate file, e.g., examples/simple_nn.py)
if __name__ == "__main__":
    import sys
    import os
    import numpy as np

    # Add the parent directory to sys.path
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    # Import the required modules
    from ftensor import nn
    from ftensor import optim
    from ftensor import data
    from ftensor import utils

    # Create a simple dataset
    X = np.random.randn(1000, 10)
    y = np.random.randint(0, 2, (1000, 1))
    dataset = data.Dataset(list(zip(X, y)))
    dataloader = data.DataLoader(dataset, batch_size=32)

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

    # Train the model
    utils.train(model, dataloader, loss_fn, optimizer, epochs=10)

    print("Training completed!")