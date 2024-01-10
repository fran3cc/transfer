# B/model.py

import torch.nn as nn
import torch.nn.functional as F

class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(NiNBlock, self).__init__()
        self.nin_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.nin_block(x)

class NiN(nn.Module):
    def __init__(self, num_classes=9):
        super(NiN, self).__init__()
        self.nin = nn.Sequential(
            NiNBlock(3, 192, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NiNBlock(192, 160, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(3, stride=2),
            NiNBlock(160, 96, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(3, stride=2),
            nn.Dropout(0.5),
            NiNBlock(96, num_classes, kernel_size=3, stride=1, padding=1)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.nin(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x
