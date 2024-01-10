import torch
from A.model import BinaryClassifierCNN
from A.utils import load_npz_data

def evaluate(model_path='A/model.pth', npz_file_path='Datasets/pneumoniamnist.npz'):
    _, _, test_loader = load_npz_data(npz_file_path)

    model = BinaryClassifierCNN()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded for evaluation.")

    correct = 0
    total = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(test_loader, 1):
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if i % 10 == 0:  # print every 10 mini-batches
                print(f'Batch {i}, Accuracy so far: {100 * correct / total:.2f}%')

    print(f'Final Accuracy: {100 * correct / total:.2f}%')

if __name__ == "__main__":
    evaluate()
