import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .utils_module import _initialize_weights

class DenseNet(nn.Module):

    def __init__(self, model_name, input_channels, num_classes, pretrained=False, subtype_head=False, cfg=None):
        super(DenseNet, self).__init__()

        backbone = pretrainedmodels.__dict__[model_name](num_classes=1000, 
                                                              pretrained='imagenet' if pretrained else None)
        self.backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.fc = nn.Linear(in_features=backbone.last_linear.in_features, out_features=num_classes)
        _initialize_weights(self.fc)

    def forward(self, x):
        x = self.backbone(x)
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x