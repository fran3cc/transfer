# B/utils.py

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

class PathMNISTDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data(data_path, batch_size=32):
    data = np.load(data_path)
    train_images, train_labels = data['train_images'], data['train_labels']
    val_images, val_labels = data['val_images'], data['val_labels']
    test_images, test_labels = data['test_images'], data['test_labels']

    transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 水平翻转
            transforms.RandomRotation(10),  # 随机旋转 ±10 度
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),  # 颜色抖动
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

    train_dataset = PathMNISTDataset(train_images, train_labels, transform)
    val_dataset = PathMNISTDataset(val_images, val_labels, transform)
    test_dataset = PathMNISTDataset(test_images, test_labels, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
