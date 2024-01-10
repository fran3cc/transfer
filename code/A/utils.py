# import numpy as np
# import torch
# from torch.utils.data import TensorDataset, DataLoader
# from torchvision import transforms

# def load_npz_data(npz_file_path, batch_size=64):
#     data = np.load(npz_file_path)
    
#     # Normalize and convert the data to PyTorch tensors
#     transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5,), (0.5,))  # Assuming the pixel values range from 0 to 1
#     ])

#     # Apply the transform and convert the numpy arrays to tensors
#     train_images_tensor = torch.stack([transform(image) for image in data['train_images']]).float()
#     val_images_tensor = torch.stack([transform(image) for image in data['val_images']]).float()
#     test_images_tensor = torch.stack([transform(image) for image in data['test_images']]).float()
    
#     # Squeeze the labels to remove the extra dimension
#     train_labels_tensor = torch.tensor(data['train_labels']).long().squeeze()
#     val_labels_tensor = torch.tensor(data['val_labels']).long().squeeze()
#     test_labels_tensor = torch.tensor(data['test_labels']).long().squeeze()

#     # Create TensorDatasets
#     train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
#     val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
#     test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

#     # Create DataLoaders
#     train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
#     val_loader = DataLoader(val_dataset, batch_size=batch_size)
#     test_loader = DataLoader(test_dataset, batch_size=batch_size)

#     return train_loader, val_loader, test_loader

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image

def load_npz_data(npz_file_path, batch_size=32):
    data = np.load(npz_file_path)
    
    # Transformations for the training data
    train_transform = transforms.Compose([
        transforms.RandomAffine(degrees=5, translate=(0.05, 0.05), scale=(0.95, 1.05)),  # Mild affine transformations
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # Normalization
    ])
    
    # Transformations for the validation and test data (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Apply transformations
    train_images_tensor = torch.stack([train_transform(Image.fromarray(img)) for img in data['train_images']])
    val_images_tensor = torch.stack([test_transform(Image.fromarray(img)) for img in data['val_images']])
    test_images_tensor = torch.stack([test_transform(Image.fromarray(img)) for img in data['test_images']])

    # Convert labels to tensors and squeeze to remove the extra dimension
    train_labels_tensor = torch.tensor(data['train_labels']).long().squeeze()
    val_labels_tensor = torch.tensor(data['val_labels']).long().squeeze()
    test_labels_tensor = torch.tensor(data['test_labels']).long().squeeze()

    # Create TensorDatasets
    train_dataset = TensorDataset(train_images_tensor, train_labels_tensor)
    val_dataset = TensorDataset(val_images_tensor, val_labels_tensor)
    test_dataset = TensorDataset(test_images_tensor, test_labels_tensor)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader
