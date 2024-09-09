import torch

class EarlyStopper():
    def __init__(self, patience=15, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, model, validation_loss, path):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            print('Saving model') 
            torch.save(model.state_dict(), f'{path}.pth')
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

def compute_validation_loss(model, data_loader, loss_fn, device='cuda'):
    val_loss = []
    with torch.no_grad():
        for valx, valy in data_loader:
            valx = valx.to(device)
            valy = valy.to(device)
            scores = model(valx)
            loss = loss_fn(scores, valy)
            val_loss.append(loss.item())
    return sum(val_loss)/len(val_loss)


def train(model, train_data_loader, val_data_loader, training_params, save_path, device='cuda'):
    epochs = training_params['epochs']
    optimizer = training_params['optimizer']
    loss_fn = training_params['loss']
    scheduler = training_params['scheduler']
    early_stopper = EarlyStopper(patience=15)
    train_loss = None
    val_loss = None
    for epoch in range(epochs):
        loss_at_epoch = []
        for x, y in tqdm(train_data_loader, desc=f"Epoch: {epoch}/{epochs} train_loss: {train_loss} val_loss: {val_loss}"):
            x = x.to(device)
            y = y.to(device)
            scores = model(x)
            loss = loss_fn(scores, y)
            loss_at_epoch.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        train_loss = sum(loss_at_epoch)/len(loss_at_epoch)
        val_loss = compute_validation_loss(model, val_data_loader, loss_fn, device=device)
        scheduler.step(val_loss)
        stop = early_stopper.early_stop(model, val_loss, save_path)
        if stop:
            print(f'Stopping early at epoch {epoch}')
            break

