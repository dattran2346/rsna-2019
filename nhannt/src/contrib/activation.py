import torch 
import torch.nn as nn 


def swish(x, inplace=False):
    if inplace:
        return x.mul_(x.sigmoid())
    else:
        return x * x.sigmoid()


class Swish(nn.Module):
    __constants__ = ['inplace']

    def __init__(self, inplace=False):
        super(Swish, self).__init__()
        self.inplace = inplace
    
    def forward(self, x):
        return swish(x, self.inplace)