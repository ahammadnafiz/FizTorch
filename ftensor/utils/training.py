# ftensor/utils/training.py
from ..nn import Module
from ..data import DataLoader
from ..optim import Optimizer
from typing import Callable

def train(model: Module, dataloader: DataLoader, loss_fn: Callable, optimizer: Optimizer, epochs: int):
    for epoch in range(epochs):
        total_loss = 0.0
        for batch_x, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Update parameters
            optimizer.step()
            
            total_loss += loss.data
        
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")