import torch
import torch.nn as nn
from torch.nn import Linear, Conv2d, BatchNorm2d
from torchvision.models import resnet18, vgg16, resnet
import torch.nn.functional as F
from utils import *
# for downloading resnet
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class ConvNet(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 3, padding=(2, 2)) 
        self.pool = nn.MaxPool2d(2, 2)  # Out: 10x114x114
        self.conv2 = nn.Conv2d(6, 10, 3)  # Out: 10x112x112
        self.conv3 = nn.Conv2d(10, 16, 3)  # Out: 16x110x110
        self.fc1 = nn.Linear(190096, 256)
        self.fc2 = nn.Linear(256, dim)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class ClassificationModel(nn.Module):
    def __init__(self, input_shape, dim=10, use_resnet=False, resnet_type='resnet18', no_fq_mask=False, freeze_mask=False, mask_initialization='ones') -> None:
        super().__init__()
        if not no_fq_mask:
            self.mask = Mask(input_shape, initialization=mask_initialization)
            if freeze_mask:
                self.mask.weights.requires_grad = False
        self.resnet_type = resnet_type
        self.no_fq_mask = no_fq_mask
        if self.resnet_type=='None':
            self.name = 'Basic'
            self.conv_model = ConvNet(dim)
        else:
            self.name = resnet_type
            resnet = torch.hub.load('pytorch/vision:v0.10.0', resnet_type, pretrained=True)
            if self.name.startswith('resnet'):
                resnet.fc = nn.Linear(512, dim)
            if self.name.startswith('vgg'):
                resnet.classifier[6] = nn.Linear(4096, dim)
            self.resnet = resnet

    def frequency_mask(self, x):
        if self.no_fq_mask:
            return x
        else:
            x = dct_2d(x)
            x = self.mask(x)  
            x = idct_2d(x)
        return x

    def convnet(self, x):
        return self.conv_model(x)

    def forward(self, x):
        x = self.frequency_mask(x)
        if self.resnet_type!='None':
            x = self.resnet(x)
        else:
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
    

