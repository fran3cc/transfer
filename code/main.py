# import A.train as train
# import A.evaluate as evaluate

# if __name__ == "__main__":
#     train.train()
#     evaluate.evaluate()

# main.py
import torch
from B.train import train_model
from B.evaluate import evaluate_model

if __name__ == "__main__":
    data_path = "./Datasets/pathmnist.npz"
    num_classes = 9  # Adjust based on your dataset
    model = train_model(data_path, num_classes, epochs=30, batch_size=64, lr=0.001)

    torch.save(model.state_dict(), './B/pathmnist_model.pth')
    model.load_state_dict(torch.load('./B/pathmnist_model.pth'))
    evaluate_model(model, data_path, batch_size=32)