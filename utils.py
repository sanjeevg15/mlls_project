from scipy.fftpack import dct, idct
import torch.nn as nn
import torch

def dct2(a):
    return dct(dct(a, axis=0, norm='ortho'), axis=1, norm='ortho' )

def idct2(a):
    return idct(idct(a, axis=0 , norm='ortho'), axis=1 , norm='ortho')

def init_mask(shape): #Can draw inspiration from Xavier Initialization
    return torch.rand(shape)
    pass

class DCT2D(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.height, self.width = input_dims
    pass

    def forward(self, x):
        x = self.dct2(x)
        return x

    def dct2(self, x):
        pass

    def idct2(self, x):
        pass

class Mask(nn.Module):
    def __init__(self, input_dims):
        super().__init__()
        self.height, self.width = input_dims
        weights = torch.Tensor(self.height, self.width)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        x = torch.mul(x, self.weights)
        return x


