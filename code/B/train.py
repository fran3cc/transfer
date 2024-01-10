# B/train.py

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.nn as nn
from .model import NiN  # Ensure this is importing the NiN model
from .utils import load_data

def train_model(data_path, num_classes, epochs=100, batch_size=128, lr=0.001, patience=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, val_loader, _ = load_data(data_path, batch_size)

    model = NiN(num_classes).to(device)  # 创建模型实例
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=patience, factor=0.1, verbose=True)

    best_val_loss = float('inf')
    no_improvement = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels.squeeze().long())
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

        # Validation step
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels.squeeze().long())
                val_loss += loss.item()
        
        
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)
        print(f'Validation Loss after Epoch {epoch+1}: {avg_val_loss}')

        # Early Stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement >= patience:
                print("Early stopping triggered")
                break

    return model
