def train(model, dataloader, loss_fn, optimizer, epochs):
    for epoch in range(epochs):
        total_loss = 0
        for batch_x, batch_y in dataloader:
            # Forward pass
            predictions = model(batch_x)
            loss = loss_fn(predictions, batch_y)

            # Backward pass
            loss.backward()

            # Update parameters
            optimizer.step()
            optimizer.zero_grad()

            total_loss += loss.data

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")