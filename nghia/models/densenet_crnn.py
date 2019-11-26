import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .densenet import DenseNet
from .utils_module import _initialize_weights


class DenseNetLSTM(nn.Module):
    def __init__(self, model_name, input_channels, num_classes, pretrained=False, subtype_head=False, cfg=None):
        super(DenseNetLSTM, self).__init__()
        self.subtype_head = subtype_head
        self.cfg = cfg

        ## Octave50 on Fold0
        pretrained_model = DenseNet(model_name, input_channels, 6, False, True, cfg)

        if pretrained:
            ckpt = torch.load(f'./weights/{cfg.TRAIN.CNN_W}.pth', "cpu")
            pretrained_model.load_state_dict(ckpt.pop('state_dict'))
        
        rfc_in_features = pretrained_model.fc.in_features
        del pretrained_model.fc
        self.backbone = pretrained_model.backbone

        if cfg.TRAIN.CNN_FROZEN:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        middle_feature = rfc_in_features // 4
        self.rfc1 = nn.Linear(rfc_in_features, middle_feature)
        self.bn1 = nn.BatchNorm1d(middle_feature, momentum=0.01)
        self.rfc2 = nn.Linear(middle_feature, middle_feature)
        self.bn2 = nn.BatchNorm1d(middle_feature, momentum=0.01)
        self.rfc3 = nn.Linear(middle_feature, 300)

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
        x = F.relu(x)
        x = F.adaptive_avg_pool2d(x, (1,1))
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