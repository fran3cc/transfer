import torch
import torch.optim as optim
import torch.nn as nn
from A.model import BinaryClassifierCNN
from A.utils import load_npz_data
import os

def train(model_path='A/model.pth', npz_file_path='Datasets/pneumoniamnist.npz'):
    train_loader, val_loader, _ = load_npz_data(npz_file_path)

    model = BinaryClassifierCNN()
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Loaded existing model.")
    else:
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)

        for epoch in range(50):  # number of epochs
            running_loss = 0.0
            for i, (inputs, labels) in enumerate(train_loader, 1):
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
                if i % 100 == 0:  # print every 100 mini-batches
                    print(f'Epoch {epoch+1}, Batch {i}, Loss: {running_loss / 100:.4f}')
                    running_loss = 0.0
            
            print(f'End of Epoch {epoch+1}, Average Loss: {running_loss / len(train_loader):.4f}')
            # Add validation step if necessary

        torch.save(model.state_dict(), model_path)
        print("Training completed and model saved.")

if __name__ == "__main__":
    train()