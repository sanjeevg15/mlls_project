import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d
from torchvision.models import resnet18, vgg16
import torch.nn.functional as F
from utils import Mask, dct2, idct2


class Model1(nn.Module):
    def __init__(self, dim=128) -> None:
        super().__init__()
        # In: 3x32x32 (CIFAR)
        self.mask = Mask((32,32))
        self.conv1 = nn.Conv2d(3, 6, 3, padding=(2, 2))  # Out: 6x32x32
        self.pool = nn.MaxPool2d(2, 2)  # Out: 6x16x16
        self.conv2 = nn.Conv2d(6, 10, 3)  # Out: 10x14x14
        self.conv3 = nn.Conv2d(10, 16, 3)  # Out: 16x12x12
        self.fc1 = nn.Linear(16*12*12, 256)
        self.fc2 = nn.Linear(256, dim)

    def apply_mask(self, x):
        x = dct2(x)
        x = self.mask(x)
        x = idct2(x)
        return x

    def convnet(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.apply_mask(x)
        x = self.convnet(x)
        return x
    
