import torch
from .utils import get_accuracy
from tqdm import tqdm

def train_densenet(model, criterion, optimizer, train_loader, valid_loader, n_epochs, device, scheduler, save_dir):
    history = {
        'train_loss': [], 
        'train_acc': [],
        'val_loss': [], 
        'val_acc': [],
        'lr': []
    }

    print(f"\n{'Epoch':^7} | {'Tr. Loss':^10} | {'Tr. Acc':^10} | "
          f"{'Val Loss':^10} | {'Val Acc':^10} | {'LR':^10}")
    print("-" * 75)

    best_val_acc = 0.0

    for epoch in range(n_epochs):
        train_losses     = 0.
        train_accuracies = 0.
        model.train()
        current_lr = optimizer.param_groups[0]['lr']

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Train]", leave=False)

        for images, labels in train_loop:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            optimizer.zero_grad()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            accuracy = get_accuracy(outputs.detach(), labels)
            train_losses += loss.item()
            train_accuracies += accuracy
            
            train_loop.set_postfix(loss=loss.item())

        avg_train_loss = train_losses / len(train_loader)
        avg_train_accuracy = train_accuracies / len(train_loader)
        
        avg_valid_loss, avg_valid_accuracy = valid_densenet(
            model, criterion, valid_loader, device, epoch, n_epochs)

        if scheduler is not None:
            scheduler.step(avg_valid_loss)

        print(f"{epoch+1:^7} | {avg_train_loss:^10.4f} | {avg_train_accuracy:^10.4f} | "
              f"{avg_valid_loss:^10.4f} | {avg_valid_accuracy:^10.4f} | {current_lr:^10.1e}")

        if avg_valid_accuracy > best_val_acc:
            best_val_acc = avg_valid_accuracy
            save_path = f"{save_dir}/best_densenet.pth"
            torch.save(model.state_dict(), save_path)
            print(f" >>> Model Saved! (Best Acc: {best_val_acc:.4f})")

        history['train_loss'].append(avg_train_loss)
        history['train_acc'].append(avg_train_accuracy)
        history['val_loss'].append(avg_valid_loss)
        history['val_acc'].append(avg_valid_accuracy)
        history['lr'].append(current_lr)

    return history


def valid_densenet(model, criterion, valid_loader, device, epoch=0, n_epochs=0):
    valid_losses = 0.
    valid_accuracies = 0.
    model.eval()
    
    valid_loop = tqdm(valid_loader, desc=f"Epoch {epoch+1}/{n_epochs} [Valid]", leave=False)
    
    with torch.no_grad():
        for images, labels in valid_loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            accuracy = get_accuracy(outputs, labels)
            
            valid_losses += loss.item()
            valid_accuracies += accuracy
            
            valid_loop.set_postfix(val_loss=loss.item())
            
    avg_valid_loss = valid_losses / len(valid_loader)
    avg_valid_accuracy = valid_accuracies / len(valid_loader)
    
    return avg_valid_loss, avg_valid_accuracy