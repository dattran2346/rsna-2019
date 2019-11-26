import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
import math
import os

import torchvision
from torchvision.models.resnet import BasicBlock, Bottleneck
import pretrainedmodels
import timm
from timm.models.conv2d_helpers import Conv2dSame
from timm.models.gen_efficientnet import InvertedResidual, _initialize_weight_goog

from contrib import WSO 
from contrib import ConvBNAct, SelfAttentionBlock
from contrib import PositionalEncoding, TransformerEncoderLayer 
from contrib import blseresnext50_32x4d_a2_b4, blseresnext101_32x4d_a2_b4


class GenericEfficientNet(nn.Module):
    """
    EfficientNet B0-B7; with auxiliary heads.
    Args:
        model_name (str): name of model to instantiate
        input_channels (int): number of input channels/ colors
        num_classes (int): number of classes for the final Linear module.
    """
    def __init__(self, model_name, input_channels, num_classes, **kwargs):
        super(GenericEfficientNet, self).__init__()
        backbone = timm.create_model(model_name, pretrained=True)
        in_features = backbone.conv_head.out_channels

        self.wso = WSO()
        self.bn0 = nn.BatchNorm2d(num_features=input_channels)
        
        self.conv_stem = backbone.conv_stem
        self.bn1 = backbone.bn1
        self.act_fn = backbone.act_fn
        for i in range(len((backbone.blocks))):
            setattr(self, "block{}".format(str(i)), backbone.blocks[i])
        self.conv_head = backbone.conv_head
        self.bn2 = backbone.bn2
        self.global_pool = backbone.global_pool
        self.drop_rate = backbone.drop_rate

        del backbone

        if input_channels != 3:
            old_conv_weight = self.conv_stem.weight
            new_conv = Conv2dSame(input_channels, 48, 3, 2, bias=False)
            with torch.no_grad(): 
                new_conv.weight = nn.Parameter(torch.stack(
                    [torch.mean(old_conv_weight, 1)] * input_channels, 1))
            self.conv_stem = new_conv

        self.fc = nn.Linear(in_features, num_classes, bias=True)
        # create self-attention block (optional)
        self.attn = kwargs["attention"]
        if self.attn:
            self.attn_block = SelfAttentionBlock(
                high_in_channels=self.block6[-1].bn3.num_features,
                low_in_channels=self.block5[-1].bn3.num_features,
                key_channels=256,
                value_channels=256,
                out_channels=self.block6[-1].bn3.num_features,
                # act="swish")
                act="relu")
        # extra heads
        self.block4_aux_fc = nn.Sequential(
            InvertedResidual(self.block4[-1].bn3.num_features, in_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, num_classes, bias=True))
        self.block5_aux_fc = nn.Sequential(
            InvertedResidual(self.block5[-1].bn3.num_features, in_features),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_features, num_classes, bias=True))
        # initialization
        nn.init.ones_(self.bn0.weight.data)
        nn.init.zeros_(self.bn0.bias.data)
        for m in [self.block4_aux_fc, self.block5_aux_fc]:
            _initialize_weight_goog(m)
        nn.init.zeros_(self.fc.bias.data)

        self.model_name = model_name

    def _features(self, x):
        x = self.wso(x)
        x = self.bn0(x)
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x = self.block0(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x); b4 = x
        x = self.block5(x); b5 = x
        x = self.block6(x)
        if self.attn:
            x = self.attn_block(x, b5)
        x = self.conv_head(x)
        x = self.bn2(x)
        x = self.act_fn(x, inplace=True)
        return b4, b5, x 
        
    def forward(self, x):
        b4, b5, x = self._features(x)
        b4_x = self.block4_aux_fc(b4)
        b5_x = self.block5_aux_fc(b5)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        if self.drop_rate > 0.:
            x = F.dropout(x, p=self.drop_rate, training=self.training)
        x = self.fc(x)
        if self.training:
            return x, b5_x, b4_x
        return (x + b5_x + b4_x) / 3.
        

class GenericEfficientNet3d(GenericEfficientNet):
    """
    3D EfficientNet.
    
    Args:
        model_name (str): name of model to instantiate
        input_channels (int): number of input channels/ colors
        num_classes (int): number of classes for the final Linear module.
    """
    def __init__(self, model_name, input_channels, num_classes, **kwargs):
        super(
            GenericEfficientNet3d,
            self).__init__(
            model_name,
            input_channels,
            num_classes,
            **kwargs)    
        in_features = self.fc.in_features
        del self.fc, self.block4_aux_fc, self.block5_aux_fc

        self.block4_aux = InvertedResidual(self.block4[-1].bn3.num_features, in_features)
        self.block5_aux = InvertedResidual(self.block5[-1].bn3.num_features, in_features)
        self.block4_decoder = RecurrentDecoder(in_features, num_classes,
            dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
            dim_ffw=kwargs["dim_ffw"], recur_type=kwargs["recur_type"], 
            num_heads=kwargs["num_heads"])
        self.block5_decoder = RecurrentDecoder(in_features, num_classes,
            dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
            dim_ffw=kwargs["dim_ffw"], recur_type=kwargs["recur_type"], 
            num_heads=kwargs["num_heads"])
        self.decoder = RecurrentDecoder(in_features, num_classes,
            dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
            dim_ffw=kwargs["dim_ffw"], recur_type=kwargs["recur_type"], 
            num_heads=kwargs["num_heads"])
        for m in [self.block4_aux, self.block4_decoder, self.block5_aux, self.block5_decoder]:
            _initialize_weight_goog(m)

    def forward(self, x, seq_len):
        b4, b5, x = self._features(x)
        b4_x = self.block4_aux(b4)
        b4_x = self.block4_decoder(b4_x, seq_len)
        b5_x = self.block5_aux(b5)
        b5_x = self.block5_decoder(b5_x, seq_len)
        x = self.decoder(x, seq_len)
        if self.training:
            return x, b5_x, b4_x
        return (x + b5_x + b4_x) / 3.


class ResNet(nn.Module):
    """
    ResNet, ResNeXt, SENet, GCNet, BigLittleNet; with auxiliary heads.

    Args:
        model_name (str): name of model to instantiate
        input_channels (int): number of input channels/ colors
        num_classes (int): number of classes for the final Linear module.
    """
    def __init__(self, model_name, input_channels, num_classes, **kwargs):
        super(ResNet, self).__init__()
        if model_name in ["resnext101_32x8d_wsl", "resnext101_32x16d_wsl"]:
            backbone = torch.hub.load(
                "facebookresearch/WSL-Images", model_name)
            in_features = backbone.fc.in_features
        elif model_name == "resnext50_32x4d":
            backbone = torchvision.models.resnext50_32x4d(pretrained=True)
            in_features = backbone.fc.in_features
        elif model_name == "blseresnext50_32x4d_a2_b4":
            backbone = blseresnext50_32x4d_a2_b4(pretrained=True)
            in_features = backbone.fc.in_features
        elif model_name == "blseresnext101_32x4d_a2_b4":
            backbone = blseresnext101_32x4d_a2_b4(pretrained=True)
            in_features = backbone.fc.in_features
        else:
            backbone = pretrainedmodels.__dict__[
                model_name](num_classes=1000, pretrained="imagenet")
            in_features = backbone.last_linear.in_features

        self.wso = WSO()
        self.bn0 = nn.BatchNorm2d(input_channels)
        
        if model_name in ['resnet18', 'resnet34', 'resnet50',
                          'resnet101', 'resnet152', 'fbresnet152',
                          'resnext50_32x4d',
                          "resnext101_32x8d_wsl", "resnext101_32x16d_wsl"]:
            layer0_modules = [
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.relu),
                ('maxpool', backbone.maxpool)
            ]
            self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
        elif model_name in ['blseresnext50_32x4d_a2_b4', 'blseresnext101_32x4d_a2_b4']:
            layer0_modules =[
                ('conv1', backbone.conv1),
                ('bn1', backbone.bn1),
                ('relu', backbone.relu)
            ]
            self.layer0 = nn.Sequential(OrderedDict(layer0_modules))
            self.b_conv0 = backbone.b_conv0
            self.bn_b0 = backbone.bn_b0
            self.l_conv0 = backbone.l_conv0
            self.bn_l0 = backbone.bn_l0
            self.relu = backbone.relu
            self.l_conv1 = backbone.l_conv1
            self.bn_l1 = backbone.bn_l1 
            self.l_conv2 = backbone.l_conv2
            self.bn_l2 = backbone.bn_l2
            self.bl_init = backbone.bl_init
            self.bn_bl_init = backbone.bn_bl_init
        else:
            self.layer0 = backbone.layer0
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        if model_name in ["se_resnext50_32x4d", "se_resnext101_32x4d",
                          "blseresnext50_32x4d_a2_b4", "blseresnext101_32x4d_a2_b4"]:
            self.groups = 32
            self.width_per_group = 4
        else:
            self.groups = backbone.groups
            self.width_per_group = backbone.base_width

        del backbone

        if input_channels != 3:
            old_conv_weight = self.layer0.conv1.weight.data
            new_conv = nn.Conv2d(input_channels, 64, kernel_size=7,
                                 stride=2, padding=3, bias=False)
            new_conv.weight.data = torch.stack(
                [torch.mean(old_conv_weight, 1)] * input_channels,
                dim=1)
            self.layer0.conv1 = new_conv

        self.fc = nn.Linear(in_features, num_classes)
        
        if model_name in ["resnet18", "resnet34"]:
            self.layer2_aux_fc = nn.Sequential(
                BasicBlock(128, in_features, 
                    downsample=nn.Sequential(
                        nn.Conv2d(128, in_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_features)),
                    norm_layer=nn.BatchNorm2d),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, num_classes, bias=True))
            self.layer3_aux_fc = nn.Sequential(
                BasicBlock(256, in_features, 
                    downsample=nn.Sequential(
                        nn.Conv2d(256, in_features, kernel_size=1, bias=False),
                        nn.BatchNorm2d(in_features)),
                    norm_layer=nn.BatchNorm2d),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(512, num_classes, bias=True))
        else:
            self.layer2_aux_fc = nn.Sequential(
                Bottleneck(512, in_features // 4, 
                           groups=self.groups, base_width=self.width_per_group, 
                           downsample=nn.Sequential(
                                nn.Conv2d(512, in_features, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_features)),
                           norm_layer=nn.BatchNorm2d),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, num_classes, bias=True))
            self.layer3_aux_fc = nn.Sequential(
                Bottleneck(1024, in_features // 4, 
                           groups=self.groups, base_width=self.width_per_group, 
                           downsample=nn.Sequential(
                                nn.Conv2d(1024, in_features, kernel_size=1, bias=False),
                                nn.BatchNorm2d(in_features)),
                           norm_layer=nn.BatchNorm2d),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(in_features, num_classes, bias=True))
        # Initialization
        nn.init.ones_(self.bn0.weight.data)
        nn.init.zeros_(self.bn0.bias.data)
        nn.init.zeros_(self.fc.bias.data)
        nn.init.zeros_(self.layer2_aux_fc[-1].bias.data)
        nn.init.zeros_(self.layer3_aux_fc[-1].bias.data)

        self.model_name = model_name

    def _features(self, x):
        x = self.wso(x)
        x = self.bn0(x) 
        x = self.layer0(x)

        if self.model_name.startswith("bl"):
            bx = self.b_conv0(x)
            bx = self.bn_b0(bx)

            lx = self.l_conv0(x)
            lx = self.bn_l0(lx)
            lx = self.relu(lx)
            lx = self.l_conv1(lx)
            lx = self.bn_l1(lx)
            lx = self.relu(lx)
            lx = self.l_conv2(lx)
            lx = self.bn_l2(lx)

            x = self.relu(bx + lx)
            x = self.bl_init(x)
            x = self.bn_bl_init(x)
            x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x); l2 = x 
        x = self.layer3(x); l3 = x 
        x = self.layer4(x)
        return l2, l3, x  

    def forward(self, x):
        l2, l3, x = self._features(x)
        l2_x = self.layer2_aux_fc(l2)
        l3_x = self.layer3_aux_fc(l3)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        if self.training:
            return x, l3_x, l2_x, 
        return (x + l3_x + l2_x) / 3.


class ResNet3d(ResNet):
    """
    3D ResNet.

    Args:
        model_name (str): name of model to instantiate
        input_channels (int): number of input channels/ colors
        num_classes (int): number of classes for the final Linear module.
    """
    def __init__(self, model_name, input_channels, num_classes, **kwargs):
        super(ResNet3d, self).__init__(model_name, input_channels, num_classes, **kwargs)
        in_features = self.fc.in_features
        del self.fc, self.layer2_aux_fc, self.layer3_aux_fc
        
        if model_name in ["resnet18", "resnet34"]:
            self.layer2_aux = BasicBlock(128, in_features, 
                downsample=nn.Sequential(
                    nn.Conv2d(128, in_features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_features)),
                norm_layer=nn.BatchNorm2d)
            self.layer3_aux = BasicBlock(256, in_features, 
                downsample=nn.Sequential(
                    nn.Conv2d(256, in_features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_features)),
                norm_layer=nn.BatchNorm2d)
        else:
            self.layer2_aux = Bottleneck(512, in_features // 4, 
                groups=self.groups, base_width=self.width_per_group, 
                downsample=nn.Sequential(
                    nn.Conv2d(512, in_features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_features)),
                norm_layer=nn.BatchNorm2d)
            self.layer3_aux = Bottleneck(1024, in_features // 4, 
                groups=self.groups, base_width=self.width_per_group, 
                downsample=nn.Sequential(
                    nn.Conv2d(1024, in_features, kernel_size=1, bias=False),
                    nn.BatchNorm2d(in_features)),
                norm_layer=nn.BatchNorm2d)

        self.layer2_decoder = RecurrentDecoder(in_features, num_classes,
            dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
            dim_ffw=kwargs["dim_ffw"], recur_type=kwargs["recur_type"], 
            num_heads=kwargs["num_heads"])
        
        self.layer3_decoder = RecurrentDecoder(in_features, num_classes,
            dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
            dim_ffw=kwargs["dim_ffw"], recur_type=kwargs["recur_type"], 
            num_heads=kwargs["num_heads"])
        
        self.decoder = RecurrentDecoder(in_features, num_classes, 
            dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
            dim_ffw=kwargs["dim_ffw"], recur_type=kwargs["recur_type"], 
            num_heads=kwargs["num_heads"])

    def forward(self, x, seq_len):
        l2, l3, x = self._features(x)
        l2_x = self.layer2_aux(l2)
        l2_x = self.layer2_decoder(l2_x, seq_len)        
        l3_x = self.layer3_aux(l3)
        l3_x = self.layer3_decoder(l3_x, seq_len)
        x = self.decoder(x, seq_len)
        if self.training:
            return x, l3_x, l2_x
        return (x + l3_x + l2_x) / 3.
        

class RecurrentDecoder(nn.Module):
    """
    N-layer "recurrent" cell (e.g. Bi/Unidirectional LSTM/GRU, Transformer) with 
    last Linear module.

    Args:
        in_features (int): the number of expected features in the inputs.
        num_classes (int): the number of predicted classes.
    """
    def __init__(self, in_features, num_classes, **kwargs):
        super(RecurrentDecoder, self).__init__()
        self.recur_type = kwargs["recur_type"]
        # Bidirectional LSTM/GRU
        if self.recur_type == "bilstm":
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=in_features // 4, 
                dropout=kwargs["dropout"], num_layers=kwargs["num_layers"], 
                bidirectional=True, batch_first=True)
            self.fc = nn.Linear(in_features // 2, num_classes)        
        if self.recur_type == "bigru":
            self.gru = nn.GRU(input_size=in_features, hidden_size=in_features // 4,
                dropout=kwargs["dropout"], num_layers=kwargs["num_layers"],
                bidirectional=True, batch_first=True)
            self.fc = nn.Linear(in_features // 2, num_classes)

        # Unidirectional LSTM/GRU
        if self.recur_type == "lstm":
            self.lstm = nn.LSTM(input_size=in_features, hidden_size=in_features // 4, 
                dropout=kwargs["dropout"], num_layers=kwargs["num_layers"],
                batch_first=True)
            self.fc = nn.Linear(in_features // 4, num_classes)
        if self.recur_type == "gru":
            self.gru = nn.GRU(input_size=in_features, hidden_size=in_features // 4, 
                dropout=kwargs["dropout"], num_layers=kwargs["num_layers"],
                batch_first=True)
            self.fc = nn.Linear(in_features // 4, num_classes)
        
        # Transformer
        # if self.recur_type == "transformer":
        #     self.pos_encoder = PositionalEncoding(d_model=in_features)
        #     self.transformer = nn.TransformerEncoder(
        #         TransformerEncoderLayer(
        #             d_model=in_features, nhead=kwargs["num_heads"], 
        #             dim_feedforward=kwargs["dim_ffw"], dropout=kwargs["dropout"]),
        #         num_layers=kwargs["num_layers"]
        #     )
        #     self.fc = nn.Linear(in_features, num_classes)
        nn.init.zeros_(self.fc.bias.data)
    
    def forward(self, x, seq_len):
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = x.reshape(-1, seq_len, x.size(-1))
        if "gru" in self.recur_type:
            x, _ = self.gru(x)
        if "lstm" in self.recur_type:
            x, _ = self.lstm(x)
        # if "transformer" in self.recur_type:
        #     x = self.pos_encoder(x)
        #     x = self.transformer(x) 
        x = self.fc(x)
        x = x.reshape(-1, x.size(-1))
        return x


def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
