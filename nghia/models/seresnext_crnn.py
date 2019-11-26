import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels

from .utils_module import _initialize_weights


class SEResNeXT50(nn.Module):
    def __init__(self, num_classes):
        super(SEResNeXT50, self).__init__()
        self.base = pretrainedmodels.se_resnext50_32x4d()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
     
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class SEResNeXT50LSTM(nn.Module):
    def __init__(self, model_name, input_channels, num_classes, pretrained=False, subtype_head=False, cfg=None):
        super(SEResNeXT50LSTM, self).__init__()
        self.cfg = cfg
        
        pretrained_model = SEResNeXT50(num_classes)

        if pretrained:
            ckpt = torch.load(f'./weights/seresnext50.pth', "cpu")
            pretrained_model.load_state_dict(ckpt)
        in_features = pretrained_model.head.in_features
        del pretrained_model.head
        self.backbone = pretrained_model.backbone

        if cfg.TRAIN.CNN_FROZEN:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        self.rfc1 = nn.Linear(in_features, 512)
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
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1)
        return x

    def forward(self, x):
        bs, ns, ch, sz, sz = x.size()
        x = x.reshape(bs * ns, ch, sz, sz)

        x = self.feature(x)
        x = x.view(x.size(0), -1)
        x = self.reduce_dim(x)
        x = x.reshape(bs, ns, -1)

        # self.lstm.flatten_parameters()
        rnn_out, (h_n, h_c) = self.lstm(x)

        x = self.fc1(rnn_out)
        x = F.relu(x)
        x = F.dropout(x, p=self.cfg.TRAIN.DROPOUT, training=self.training)
        x = self.fc2(x)
        x = x.reshape(bs * ns, -1)

        return x