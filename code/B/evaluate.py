# B/evaluate.py

import torch
from .utils import load_data

def evaluate_model(model, data_path, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, test_loader = load_data(data_path, batch_size)

    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.squeeze()).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy on test images: {accuracy}%')
