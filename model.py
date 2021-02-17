import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.relu = nn.LeakyReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels, affine=True)

        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels, affine=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.maxpool(out)

        return out


class TinyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.b1 = Block(3, 12)
        self.b2 = Block(12, 18)
        self.b3 = Block(18, 36)
        self.b4 = Block(36, 48)
        self.b5 = Block(48, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=2)
        self.avgpool = nn.AdaptiveAvgPool2d((4, 6))

        self.fc_bbox = nn.Sequential(
            nn.Linear(24 * 64, 64),
            nn.LeakyReLU(inplace=True),
            nn.Linear(64, 16),
            nn.LeakyReLU(inplace=True),
            nn.Linear(16, 4)
        )

        self.fc_logits = nn.Linear(24 * 64, 2)

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)
        x = self.avgpool(x)
        x = torch.flatten(x, start_dim=1)
        logits = self.fc_logits(x)
        bbox = self.fc_bbox(x)
        return logits, bbox
