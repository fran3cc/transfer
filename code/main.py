# import A.train as train
# import A.evaluate as evaluate

# if __name__ == "__main__":
#     train.train()
#     evaluate.evaluate()

# main.py

from B.train import train_model
from B.evaluate import evaluate_model
import torch

if __name__ == "__main__":
    data_path = "./Datasets/pathmnist.npz"
    model = train_model(data_path, epochs=30, batch_size=32, lr=0.001)

    # Saving the model
    torch.save(model.state_dict(), './B/pathmnist_model.pth')

    # Load model for evaluation
    model.load_state_dict(torch.load('./B/pathmnist_model.pth'))
    evaluate_model(model, data_path, batch_size=32)
