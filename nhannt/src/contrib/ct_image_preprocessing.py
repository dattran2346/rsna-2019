from collections import OrderedDict
import math 
import torch 
import torch.nn as nn 


def get_init_conv_params_sigmoid(ww, wl, smooth=1., upbound_value=255.):
    """
    Source: https://github.com/MGH-LMIC/windows_optimization/blob/master/functions.py
    """
    w = 2./ww * math.log(upbound_value/smooth - 1.)
    b = -2.*wl/ww * math.log(upbound_value/smooth - 1.)
    return (w, b)


class WSO(nn.Module):
    """
    Window settings optimization from 
    `"Practical Window Setting Optimization for Medical Image Deep Learning" <https://arxiv.org/pdf/1812.00572.pdf>`_
    """
    def __init__(self, windows=OrderedDict({
                    'brain': {'W': 80, 'L': 40},
                    'subdural': {'W': 215, 'L': 75},
                    'bony': {'W': 2800, 'L': 600},
                    'tissue': {'W': 375, 'L': 40},
                    }),
                 U=255., eps=1.):
        super(WSO, self).__init__()
        self.windows = windows
        self.U = U
        self.eps = eps
        self.conv1x1 = nn.Conv2d(1, len(windows), kernel_size=1, stride=1, padding=0)
        
        weight, bias = self._get_window_params()
        self.conv1x1.weight.data = weight[:, None, None, None]
        self.conv1x1.bias.data = bias

    def _get_window_params(self):
        weight = []
        bias = []
        for _, window in self.windows.items():
            ww, wl = window["W"], window["L"]
            w, b = get_init_conv_params_sigmoid(ww, wl, self.eps, self.U)
            weight.append(w)
            bias.append(b)
        weight = torch.as_tensor(weight)
        bias = torch.as_tensor(bias)
        return weight, bias

    def forward(self, x):
        x = self.conv1x1(x)
        x = torch.sigmoid(x)
        x = x.mul(self.U)
        return x