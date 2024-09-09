# Importing PyTorch library
import torch

# Class for early stopping mechanism
class EarlyStopper():
    def __init__(self, patience=15, min_delta=0):
        # Initializing patience for early stopping and minimum improvement delta
        self.patience = patience
        self.min_delta = min_delta
        # Counter to track how many consecutive epochs without improvement
        self.counter = 0
        # Setting the initial minimum validation loss to infinity
        self.min_validation_loss = float('inf')

    # Function to check if training should stop early based on validation loss
    def early_stop(self, model, validation_loss, path):
        # If current validation loss is better than the minimum seen so far
        if validation_loss < self.min_validation_loss:
            # Update minimum validation loss to the current loss
            self.min_validation_loss = validation_loss
            # Reset the counter as validation loss has improved
            self.counter = 0
            print('Saving model') 
            # Save the model's state to the specified path
            torch.save(model.state_dict(), f'{path}.pth')
        # If validation loss has not improved by more than the min_delta
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            # Increment the counter as the loss hasn't improved
            self.counter += 1
            # If the counter exceeds patience, trigger early stopping
            if self.counter >= self.patience:
                return True
        # Return False to indicate training can continue
        return False

# Function to compute the average validation loss
def compute_validation_loss(model, data_loader, loss_fn, device='cuda'):
    # List to accumulate validation losses
    val_loss = []
    # Disable gradient computation during validation
    with torch.no_grad():
        # Iterate through the validation dataset
        for valx, valy in data_loader:
            # Move input data and labels to the specified device (GPU/CPU)
            valx = valx.to(device)
            valy = valy.to(device)
            # Forward pass: get the model's predictions
            scores = model(valx)
            # Compute loss between predictions and actual labels
            loss = loss_fn(scores, valy)
            # Append the computed loss to the list
            val_loss.append(loss.item())
    # Return the average validation loss over the entire validation dataset
    return sum(val_loss)/len(val_loss)

# Function to train the model with early stopping and validation tracking
def train(model, train_data_loader, val_data_loader, training_params, save_path, device='cuda'):
    # Extract training parameters
    epochs = training_params['epochs']            # Total number of training epochs
    optimizer = training_params['optimizer']      # Optimizer for gradient updates
    loss_fn = training_params['loss']             # Loss function used during training
    scheduler = training_params['scheduler']      # Learning rate scheduler
    # Initialize the EarlyStopper class with a patience of 15 epochs
    early_stopper = EarlyStopper(patience=15)
    # Initialize train and validation loss for displaying progress
    train_loss = None
    val_loss = None
    # Loop through each epoch
    for epoch in range(epochs):
        # List to collect the loss for each batch in the current epoch
        loss_at_epoch = []
        # Loop through the training data batches
        for x, y in tqdm(train_data_loader, desc=f"Epoch: {epoch}/{epochs} train_loss: {train_loss} val_loss: {val_loss}"):
            # Move inputs and labels to the specified device (GPU/CPU)
            x = x.to(device)
            y = y.to(device)
            # Forward pass: get the model's predictions
            scores = model(x)
            # Compute the loss between predictions and actual labels
            loss = loss_fn(scores, y)
            # Add the current batch loss to the list
            loss_at_epoch.append(loss.item())
            # Zero the gradients for the optimizer
            optimizer.zero_grad()
            # Backward pass: compute gradients
            loss.backward()
            # Update model weights using the optimizer
            optimizer.step()
        # Calculate the average training loss for the epoch
        train_loss = sum(loss_at_epoch) / len(loss_at_epoch)
        # Compute the validation loss for the epoch
        val_loss = compute_validation_loss(model, val_data_loader, loss_fn, device=device)
        # Update the learning rate based on validation loss using scheduler
        scheduler.step(val_loss)
        # Check if early stopping criteria are met (based on validation loss)
        stop = early_stopper.early_stop(model, val_loss, save_path)
        if stop:
            print(f'Stopping early at epoch {epoch}')
            break
