# import torch.nn as nn
# import torch.nn.functional as F

# class BinaryClassifierCNN(nn.Module):
#     def __init__(self):
#         super(BinaryClassifierCNN, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(32)  # Batch Normalization after first convolution
#         self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(64)  # Batch Normalization after second convolution
#         self.fc1 = nn.Linear(64 * 7 * 7, 128)
#         self.fc2 = nn.Linear(128, 64)
#         self.fc3 = nn.Linear(64, 2)
#         self.dropout = nn.Dropout(0.5)  # Dropout layer with a dropout rate of 0.5

#     def forward(self, x):
#         x = self.pool(F.relu(self.bn1(self.conv1(x))))  # Apply batch normalization after ReLU
#         x = self.pool(F.relu(self.bn2(self.conv2(x))))  # Apply batch normalization after ReLU
#         x = x.view(-1, 64 * 7 * 7)
#         x = F.relu(self.fc1(x))
#         x = self.dropout(x)  # Apply dropout
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x



# # import torch.nn as nn
# # import torchvision.models as models
# # from torchvision.models.resnet import ResNet18_Weights

# # class BinaryClassifierCNN(nn.Module):
# #     def __init__(self):
# #         super(BinaryClassifierCNN, self).__init__()
# #         # Load a pre-trained ResNet18 model with updated syntax
# #         self.resnet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        
# #         # Modify the first convolution layer
# #         self.resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)

# #         # Replace the final fully connected layer
# #         num_features = self.resnet.fc.in_features
# #         self.resnet.fc = nn.Linear(num_features, 2)

# #     def forward(self, x):
# #         return self.resnet(x)

import torch.nn as nn
import torch.nn.functional as F

class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # New layer
        self.bn3 = nn.BatchNorm2d(128)  # New layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(128 * 3 * 3, 256)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  # Additional layer
        x = x.view(-1, 128 * 3 * 3)  # Adjust the size here
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

