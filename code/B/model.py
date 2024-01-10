# B/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(NiNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class NiN(nn.Module):
    def __init__(self, num_classes=9):
        super(NiN, self).__init__()
        self.nin_block1 = NiNBlock(3, 32, kernel_size=5, stride=1, padding=2)
        self.nin_block2 = NiNBlock(32, 64, kernel_size=5, stride=1, padding=2)
        self.nin_block3 = NiNBlock(64, 128, kernel_size=3, stride=1, padding=1)
        self.dropout = nn.Dropout(0.5)
        self.final_block = NiNBlock(128, num_classes, kernel_size=3, stride=1, padding=1)
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.nin_block1(x)
        x = self.nin_block2(x)
        x = self.nin_block3(x)
        x = self.dropout(x)
        x = self.final_block(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x
