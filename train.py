import torch
import torch.nn as nn
import torch.optim as optim
from src import DenseNet
from src import get_loaders
from src import train_densenet
from src import plot_training_results

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

BATCH_SIZE = 32
LEARNING_RATE = 1e-4
EPOCHS = 30

train_loader, valid_loader, classes = get_loaders(batch_size=BATCH_SIZE)
model = DenseNet(num_classes=len(classes)).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

if __name__ == "__main__":
    history = train_densenet(
        model=model, 
        criterion=criterion, 
        optimizer=optimizer, 
        train_loader=train_loader, 
        valid_loader=valid_loader, 
        n_epochs=EPOCHS, 
        device=device, 
        scheduler=scheduler,
        save_dir='models'
    )

    plot_training_results(history)