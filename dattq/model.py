import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import os
import pretrainedmodels
from functools import partial
from cbam import CBAM
from global_context import ContextBlock2d
from ssa import SimpleSelfAttention
from scse import ChannelSELayer, SpatialSELayer, ChannelSpatialSELayer
from layer_utils import modify_resnet_layer0, modify_resnet_bottleneck, modify_last_pool

from utils import load_state_dict
from util_modules import Flatten, flatten, AdaptiveConcatPool2d, adaptive_concat_pool2d, \
    conv3x3, conv_with_kaiming_uniform, fc_with_kaiming_uniform
from layers import GeM
import torchvision

def zero_out_bn(bn):
    nn.init.constant_(bn.weight, 0)

class Noop(nn.Module):

    def forward(self, x):
        return x

# pretrained model
from torchvision.models import resnet50, resnet101, resnet152
from torchvision.models import resnet18, resnet34
from torchvision.models import resnext50_32x4d

# se net
from pretrainedmodels.models import se_resnet50, se_resnet101, se_resnet152, se_resnext50_32x4d, se_resnext101_32x4d

# imagenet wsl model
from resnext_wsl import resnext101_32x8d_wsl, resnext101_32x16d_wsl, resnext101_32x32d_wsl, resnext101_32x48d_wsl


torchvision_model = {
    'resnet18': partial(resnet18, pretrained=True),
    'resnet34': partial(resnet34, pretrained=True),

    'resnet50': partial(resnet50, pretrained=True),
    'resnet101': partial(resnet101, pretrained=True),
    'resnet152': partial(resnet152, pretrained=True),

    'resnext50_32x4d': partial(resnext50_32x4d, pretrained=True),
    'resnext101_32x8d': resnext101_32x8d_wsl,
    'resnext101_32x16d': resnext101_32x16d_wsl,
    'resnext101_32x32d': resnext101_32x32d_wsl,
    'resnext101_32x48d': resnext101_32x48d_wsl,

}

cadene_model = {
    'se_resnet50': se_resnet50,
    'se_resnet101': se_resnet101,
    'se_resnet152': se_resnet152,
    'se_resnext50_32x4d': se_resnext50_32x4d,
    'se_resnext101_32x4d': se_resnext101_32x4d,
}

class ResNet(nn.Module):
    def __init__(self, args):
        super(ResNet, self).__init__()
        att = args.att
        self.args = args

        if att == 'se':
            # load resnet with se attention module
            model_name = f'se_{args.backbone}'
            self.backbone = cadene_model[model_name](num_classes=1000, pretrained='imagenet')
            modify_last_pool(self.backbone)

        elif args.backbone in torchvision_model.keys():
            # load resnet family model
            model_name = args.backbone
            self.backbone = torchvision_model[model_name]()
            modify_resnet_layer0(self.backbone, args)

        # add attention module
        if att == 'cbam':
            modify_resnet_bottleneck(self.backbone, CBAM)
        elif att == 'ssa':
            # washout pretrained weight of bn3 as suggest in github repo
            for m in self.backbone.modules():
                if isinstance(m, torchvision.models.resnet.Bottleneck):
                    zero_out_bn(m.bn3)
            modify_resnet_bottleneck(self.backbone, SimpleSelfAttention)
        elif att == 'gc':
            modify_resnet_bottleneck(self.backbone, ContextBlock2d)
        elif att == 'cse':
            modify_resnet_bottleneck(self.backbone, ChannelSELayer)
        elif att == 'sse':
            pass
            # se net
            # modify_resnet_bottleneck(self.backbone, SpatialSELayer)
        elif att == 'scse':
            modify_resnet_bottleneck(self.backbone, ChannelSpatialSELayer)
        else:
            print("No attention module")


        # global pooling layer
        self.backbone.avg_pool = nn.AdaptiveAvgPool2d(output_size=1)
        # self.backbone.avg_pool = GeM(p=1)

        # last fc
        ## cut resnet layer
        # if args.backbone in ['resnet18', 'resnet34', 'resnet50', 'resnet101'] and args.cut_block:
        #     del self.backbone.layer4[-1]
        
        # if args.backbone == 'resnet50':
        #     # resnet18: 2, 2, 2, 2
        #     # resnet50: 3, 4, 6, 3
        #     del self.backbone.layer1[2:]
        #     del self.backbone.layer2[2:]
        #     del self.backbone.layer3[2:]
        #     del self.backbone.layer4[2:]

        if self.args.input_level == 'per-slice':
            out_features = 512 if model_name in ['resnet18', 'resnet34'] else 2048
            self.fc = nn.Linear(out_features, args.nclasses)

    def forward(self, x1):
        # self.bn = # let model learn the normalization by itself ?? init w=1, b=0
        x1 = self.backbone.layer0(x1)
        x1 = self.backbone.layer1(x1)
        x1 = self.backbone.layer2(x1)
        x1 = self.backbone.layer3(x1)
        x1 = self.backbone.layer4(x1)
       
        if self.args.conv_lstm:
            # return bs*nslices, nfeatures, h//32, w//32
            return x1
        else:
            # return bs*nslices, nfeatures
            x1 = self.backbone.avg_pool(x1)
            x = flatten(x1) # no flatten for conv lstm
            return x


        # if self.args.input_level == 'per-slice':
        #     x = self.fc(x)
        #     return x
        # else:
        #     return x


# FC LSTM Decoder
from collections import OrderedDict
class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(input_size=args.nfeatures, hidden_size=args.nfeatures//4, num_layers=2, batch_first=True, dropout=0) # drop not good
        self.fc = nn.Linear(args.nfeatures//4, args.nclasses)

        ## these init is also not good :))
        # # ### Zero init
        # # bias_ih_l[k] (b_ii|b_if|b_ig|b_io)`
        # # bias_hh_l[k] (b_hi|b_hf|b_hg|b_ho)
        # self.lstm.bias_ih_l0.data.zero_()
        # self.lstm.bias_hh_l0.data.zero_()
        # self.lstm.bias_ih_l1.data.zero_()
        # self.lstm.bias_hh_l1.data.zero_()
        # self.fc.bias.data.zero_()

        # ### Set forget bias to 1, suggest by `An Empirical Exploration of Recurrent Network Architectures`
        # self.lstm.bias_ih_l0.data[512*1:512*2].fill_(1.)
        # self.lstm.bias_hh_l0.data[512*1:512*2].fill_(1.)
        # self.lstm.bias_ih_l1.data[512*1:512*2].fill_(1.)
        # self.lstm.bias_hh_l1.data[512*1:512*2].fill_(1.)

        # ### xavier init, suggest by https://danijar.com/language-modeling-with-layer-norm-and-gru/
        # ### default init is uniform distribution
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0.data)
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0.data)
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l1.data)
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l1.data)

    def forward(self, x):
        # input: (args.bs*args.nslices, args.nfeatures)
        # output: (args.bs*args.nslices, args.nclasses)
        # view dont support back prop, reshape ok ??
        x, _ = self.lstm(x) # x (bs, nslices, nfeatures)
        x = self.fc(x)
        return  x

#################################################
# Convolutional LSTM
from convlstm import ConvLSTM

class ConvDecoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lstm = ConvLSTM(input_size=(args.image_size//32, args.image_size//32), 
                        kernel_size=(3, 3), input_dim=args.nfeatures, hidden_dim=args.nfeatures//4, num_layers=2, 
                        batch_first=True, return_all_layers=False)
        self.fc = nn.Linear(args.nfeatures//4, args.nclasses)
    
    def forward(self, x):
        # x: (bs, nslices, nfeatures, h, w)
        layer_outputs, _ = self.lstm(x)
        x = layer_outputs[0] # -> (bs, nslices, nfeatues//4, h, w)
        x = x.mean(dim=(3, 4)) # -> adaptive average pool in spatital dimension
        return self.fc(x) # -> bs, nslices, nclasses