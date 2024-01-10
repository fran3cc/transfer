import torch
import torch.nn as nn
import torch.nn.functional as F

class NiNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(NiNBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class NiN(nn.Module):
    def __init__(self, num_classes=9):
        super(NiN, self).__init__()
        self.nin_block1 = NiNBlock(3, 48, kernel_size=5, stride=1, padding=2)
        self.nin_block2 = NiNBlock(48, 96, kernel_size=5, stride=1, padding=2)
        self.nin_block3 = NiNBlock(96, 192, kernel_size=3, stride=1, padding=1)
        self.nin_block4 = NiNBlock(192, 192, kernel_size=3, stride=1, padding=1)  # New block
         # 添加一个额外的 NiNBlock
        self.nin_block5 = NiNBlock(192, 256, kernel_size=3, stride=1, padding=1)  # 新增块

        # 其他层保持不变
        self.dropout = nn.Dropout(0.5)
        self.final_block = NiNBlock(256, num_classes, kernel_size=3, stride=1, padding=1)  # 注意调整输入通道数
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        # Residual connections
        self.residual1 = nn.Conv2d(3, 48, kernel_size=1, stride=1, padding=0)
        self.residual2 = nn.Conv2d(48, 96, kernel_size=1, stride=1, padding=0)
        self.residual3 = nn.Conv2d(96, 192, kernel_size=1, stride=1, padding=0)
        self.residual4 = nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.residual5 = nn.Conv2d(192, 256, kernel_size=1, stride=1, padding=0)  # 对应的残差连接

    def forward(self, x):
        x = self.nin_block1(x) + self.residual1(x)
        x = self.nin_block2(x) + self.residual2(x)
        x = self.nin_block3(x) + self.residual3(x)
        x = self.nin_block4(x) + self.residual4(x)
        x = self.nin_block5(x) + self.residual5(x)
        x = self.dropout(x)
        x = self.final_block(x)
        x = self.global_avg_pool(x)
        x = torch.flatten(x, 1)
        return x