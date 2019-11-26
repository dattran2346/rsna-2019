import torchvision
import pretrainedmodels
from torch import nn
from collections import OrderedDict
import types

def modify_resnet_bottleneck(model, att_module):
    def bottleneck_forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out = self.se_module(out) + residual
        out = self.relu(out)

        return out
        
        
    for m in model.modules():
        if isinstance(m, torchvision.models.resnet.Bottleneck):
            # resnet50-101-152, resnext50_32x4d, resnext101_32x8d, wide_resnet50, wide_resnet101
            # add attention module
            out_channels = m.conv3.out_channels
            m.se_module = att_module(out_channels)
            
            # modify forward method
            m.forward = types.MethodType(bottleneck_forward, m)
        elif isinstance(m, pretrainedmodels.models.senet.Bottleneck):
            # cadene
            # se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d
            # change attention module
            out_channels = m.conv3.out_channels
            m.se_module = att_module(out_channels)


def modify_last_pool(model):
    if not isinstance(model, pretrainedmodels.models.senet.SENet): return
        
    model.avg_pool = nn.AdaptiveAvgPool2d((1,1))


from timm.models.conv2d_helpers import Conv2dSame
import torch

def modify_resnet_layer0(model, args):
    if not isinstance(model, torchvision.models.resnet.ResNet): return
    
    # handle 6 input channel image
    if args.mix_window == 6:
        old_conv = model.conv1
        old_conv_weight = old_conv.weight
        new_conv = Conv2dSame(6, 64, 3, 2, bias=False)
        with torch.no_grad():
            new_conv.weight = nn.Parameter(torch.stack([torch.mean(old_conv_weight, 1)] * 6, 1))
        conv1 = new_conv
    else:
        conv1 = model.conv1
    

    # group stem
    layer0_modules = [
            ('conv1', conv1),
            ('bn1', model.bn1),
            ('relu', model.relu),
            ('maxpool', model.maxpool)
        ]
    model.layer0 = nn.Sequential(OrderedDict(layer0_modules))
    
    # delete unused layer
    del model.conv1
    del model.bn1
    del model.relu
    del model.maxpool
    
    # modify feature method from cadene
    def features(self, input):
        x = self.layer0(input)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    model.features = types.MethodType(features, model)
