from scipy.fftpack import dct, idct
import torch.nn as nn
import torch
import numpy as np

def dct1(x):
    """
    PyTorch
    Discrete Cosine Transform, Type I
    Equivalent to: scipy.fftpack.dct(x, type=1)
    :param x (torch.tensor): the input signal
    :return: the DCT-I of the signal over the last dimension
    """
    x_shape = x.shape
    x = x.view(-1, x_shape[-1])
    return torch.fft.rfft(torch.cat([x, x.flip([1])[:, 1:-1]], dim=1), dim=1).real.view(*x_shape)

def idct1(X):
    '''
    PyTorch
    The inverse of DCT-I, which is just a scaled DCT-I
    Our definition if idct1 is such that idct1(dct1(x)) == x
    Equivalent to: scipy.fftpack.idct(x, type=1) upto a scaling constant

    :param X (torch.tensor): the input signal
    :return: the inverse DCT-I of the signal over the last dimension
    '''
    n = X.shape[-1]
    return dct1(X) / (2 * (n - 1))

def dct2(x, norm=None):
    '''
        Discrete Cosine Transform, Type II (a.k.a. the DCT)
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param x (torch.tensor): the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
    '''
    x_shape = x.shape
    N = x_shape[-1]
    x = x.contiguous().view(-1, N)

    v = torch.cat([x[:, ::2], x[:, 1::2].flip([1])], dim=1)
    # v = torch.cat([x, x.flip([1])], dim=1)
    Vc = torch.fft.fft(v)

    k = - torch.arange(N, dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V = Vc.real * W_r - Vc.imag * W_i

    if norm == 'ortho':
        V[:, 0] /= np.sqrt(N) * 2
        V[:, 1:] /= np.sqrt(N / 2) * 2
    V = 2*V.view(*x_shape)
    return V

def idct2(x, norm=None):
    '''
    The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct(dct(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X (torch.tensor): the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the inverse DCT-II of the signal over the last dimension

    '''
    x_shape = x.shape
    N = x_shape[-1]

    X_v = x.contiguous().view(-1, x_shape[-1]) / 2

    if norm == 'ortho':
        X_v[:, 0] *= np.sqrt(N) * 2
        X_v[:, 1:] *= np.sqrt(N / 2) * 2

    k = torch.arange(x_shape[-1], dtype=x.dtype, device=x.device)[None, :] * np.pi / (2 * N)
    W_r = torch.cos(k)
    W_i = torch.sin(k)

    V_t_r = X_v
    V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

    V_r = V_t_r * W_r - V_t_i * W_i
    V_i = V_t_r * W_i + V_t_i * W_r

    V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)
    V = torch.view_as_complex(V)
    v = torch.fft.ifft(V).real
    y = v.new_zeros(v.shape)
    y[:, ::2] += v[:, :N - (N // 2)]
    y[:, 1::2] += v.flip([1])[:, :N // 2]

    return y.view(*x_shape)

def dct_2d(x, norm='ortho'):
    """
    2-dimentional Discrete Cosine Transform, Type II (a.k.a. the DCT)
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param x (torch.tensor): the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    X1 = dct2(x, norm=norm)
    X2 = dct2(X1.transpose(-1, -2), norm=norm)
    return X2.transpose(-1, -2)

def idct_2d(X, norm='ortho'):
    """
    The inverse to 2D DCT-II, which is a scaled Discrete Cosine Transform, Type III
    Our definition of idct is that idct_2d(dct_2d(x)) == x
    For the meaning of the parameter `norm`, see:
    https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
    :param X (torch.tensor): the input signal
    :param norm: the normalization, None or 'ortho'
    :return: the DCT-II of the signal over the last 2 dimensions
    """
    x1 = idct2(X, norm=norm)
    x2 = idct2(x1.transpose(-1, -2), norm=norm)
    return x2.transpose(-1, -2)

def dct2d(x):
    '''
        Scipy 2-D DCT
    '''
    return dct(dct(x, axis=0, norm='ortho'), axis=1, norm='ortho' )

def idct2d(x):
    '''
        Scipy 2-D IDCT
    '''
    return idct(idct(x, axis=0 , norm='ortho'), axis=1 , norm='ortho')

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
    def __init__(self, input_dims, initialization='ones'):
        super().__init__()
        if initialization=='ones':
            weights = torch.ones(input_dims)
        elif initialization=='xavier':
            pass
        elif initialization=='random_normal':
            nn.init.normal_(self.weights)
        self.weights = nn.Parameter(weights)

    def forward(self, x):
        # print('X Shape: ', x.shape)
        # print('Weigths Shape: ', self.weights.shape)
        x = x * torch.sigmoid(self.weights.unsqueeze(0))
        return x


# Compare against scipy implementations of the above functions
if __name__ == '__main__':
    x = torch.rand(5,5)
