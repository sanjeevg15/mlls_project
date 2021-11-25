import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d
from torchvision.models import resnet18, vgg16, resnet
import torch.nn.functional as F
from utils import *


class Model1(nn.Module):
    def __init__(self, input_shape, dim=128) -> None:
        super().__init__()
        # In: 3x32x32 (CIFAR)
        self.mask = Mask(input_shape)
        self.conv1 = nn.Conv2d(3, 6, 3, padding=(2, 2))  # Out: 6x32x32
        self.pool = nn.MaxPool2d(2, 2)  # Out: 6x16x16
        self.conv2 = nn.Conv2d(6, 10, 3)  # Out: 10x14x14
        self.conv3 = nn.Conv2d(10, 16, 3)  # Out: 16x13x13
        self.fc1 = nn.Linear(16*13*13, 256)
        self.fc2 = nn.Linear(256, dim)

    def apply_mask1(self, x):
        # print(x.shape)
        x = dct_2d(x)
        # print(x.shape)
        x = self.mask(x)  
        # print(x.shape)
        x = idct_2d(x)
        return x

    def apply_mask2(self, x):
        return self.mask(x)

    def convnet(self, x):
        # print('ConvNet Input Shape: ',  x.shape)
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        # print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def forward(self, x):
        x = self.apply_mask1(x)
        x = self.convnet(x)
        return x
    
class SegmentationModel(nn.Module):
    def __init__(self, input_shape, backbone='fcn_resnet101') -> None:
        super().__init__()
        self.backbone = torch.hub.load('pytorch/vision:v0.10.0', backbone, pretrained=True)
        self.mask = Mask(input_shape)

    def frequency_mask(self, x):
        x = dct_2d(x)
        x = self.mask(x)  
        x = idct_2d(x)
        return x

    def convnet(self, x):
        x = self.backbone(x)
        return x

    def forward(self, x):
        x = self.frequency_mask(x)
        x = self.convnet(x)
        return x
    

