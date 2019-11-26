import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable

torch_ver = torch.__version__[:3]

__all__ = ['GlobalAvgPool2d', 'GramMatrix',
           'View', 'Sum', 'Mean', 'Normalize', 'ConcurrentModule',
           'PyramidPooling']

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        """Global average pooling over the input's spatial dimensions"""
        super(GlobalAvgPool2d, self).__init__()

    def forward(self, inputs):
        return F.adaptive_avg_pool2d(inputs, 1).view(inputs.size(0), -1)

def reg_classifier(model):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if 'head' in name:
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp(model, model_source_weights):
    fea_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if 'head' not in name:
            fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss