import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .octconv import oct_resnet50
from .utils_module import _initialize_weights, _freeze_weights

# Pretrained OctRes5
from .octave_resnet import OctResNet50 

class OctResHybridLSTM(nn.Module):
    def __init__(self, model_name, input_channels, num_classes, pretrained=False, subtype_head=False, cfg=None):
        super(OctResHybridLSTM, self).__init__()
        self.subtype_head = subtype_head
        self.cfg = cfg

        pretrained_model = OctResNet50(model_name, input_channels, 6, False, True, cfg)

        if pretrained:
            ckpt = torch.load(f'./weights/{cfg.TRAIN.CNN_W}.pth', "cpu")
            pretrained_model.load_state_dict(ckpt.pop('state_dict'))
        
        self.fc_sub = pretrained_model.fc
        self.fc_any = pretrained_model.fc_any

        del pretrained_model.fc
        del pretrained_model.fc_any
        self.backbone = pretrained_model.backbone

        
        self.rfc1 = nn.Linear(2048, 512)
        self.bn1 = nn.BatchNorm1d(512, momentum=0.01)
        self.rfc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512, momentum=0.01)
        self.rfc3 = nn.Linear(512, 300)

        self.lstm = nn.LSTM(
            input_size=300,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            bidirectional=cfg.TRAIN.BIDIRECTIONAL
        )

        if cfg.TRAIN.CNN_FROZEN:
            for param in self.backbone.parameters():
                param.requires_grad = False
            _freeze_weights(self.rfc1)
            _freeze_weights(self.rfc2)
            _freeze_weights(self.rfc3)
            _freeze_weights(self.lstm)


        if cfg.TRAIN.BIDIRECTIONAL:
            self.fc1 = nn.Linear(256*2, 128)
        else:
            self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, out_features=num_classes)

    def reduce_dim(self, x):
        x = self.bn1(self.rfc1(x))
        x = F.relu(x)
        x = self.bn2(self.rfc2(x))
        x = F.relu(x)
        x = F.dropout(x, p=self.cfg.TRAIN.DROPOUT, training=self.training)
        x = self.rfc3(x)
        return x

    def feature(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x_h, x_l = self.backbone.layer1(x)
        x_h, x_l = self.backbone.layer2((x_h,x_l))
        x_h, x_l = self.backbone.layer3((x_h,x_l))
        x_h_3 = self.backbone.avgpool(x_h)
        x_h, x_l = self.backbone.layer4((x_h,x_l))
        x_h_4 = self.backbone.avgpool(x_h)

        return x_h_4, x_h_3


    def forward(self, x):
        bs, ns, ch, sz, sz = x.size()
        x = x.reshape(bs * ns, ch, sz, sz)

        x, x_any = self.feature(x)
        x = x.view(x.size(0), -1)
        x_any = x_any.view(x_any.size(0), -1)

        # CNN  
        x_2d_any = self.fc_any(x_any)
        x_2d_sub = self.fc_sub(x)
        x_2d = torch.cat((x_2d_any, x_2d_sub), 1)

        # RNN
        x = self.reduce_dim(x)
        x = x.reshape(bs, ns, -1)

        rnn_out, (h_n, h_c) = self.lstm(x)

        x = self.fc1(rnn_out)
        x = F.relu(x)
        x = F.dropout(x, p=self.cfg.TRAIN.DROPOUT, training=self.training)
        x = self.fc2(x)
        x = x.reshape(bs * ns, -1)

        if self.training:
            return x, x_2d

        return (x + x_2d) / 2.