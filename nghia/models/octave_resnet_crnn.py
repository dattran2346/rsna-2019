import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .octconv import oct_resnet50
from .utils_module import _initialize_weights

# Pretrained OctRes5
from .octave_resnet import OctResNet50 

class OctResLSTM(nn.Module):
    def __init__(self, model_name, input_channels, num_classes, pretrained=False, subtype_head=False, cfg=None):
        super(OctResLSTM, self).__init__()
        self.subtype_head = subtype_head
        self.cfg = cfg
        
        ## Octave50 pretrained ImageNet
        # self.backbone = oct_resnet50()

        # if pretrained:
        #     package_directory = os.path.dirname(os.path.abspath(__file__))
        #     pretrained_path = os.path.join(package_directory, 'octconv/oct_resnet50_cosine.pth')
        #     self.backbone.load_state_dict(torch.load(pretrained_path))
        #     print(f"loaded pretrained at {pretrained_path} !")
    
        # del self.backbone.fc

        ## Octave50 on Fold0
        pretrained_model = OctResNet50(model_name, input_channels, 6, False, True, cfg)

        if pretrained:
            ckpt = torch.load(f'./weights/{cfg.TRAIN.CNN_W}.pth', "cpu")
            pretrained_model.load_state_dict(ckpt.pop('state_dict'))
            print(f"Loaded {cfg.TRAIN.CNN_W} (best: {ckpt['best_metric']})")
        
        del pretrained_model.fc
        del pretrained_model.fc_any
        self.backbone = pretrained_model.backbone

        if cfg.TRAIN.CNN_FROZEN:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
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
        x_h, x_l = self.backbone.layer4((x_h,x_l))
        x = self.backbone.avgpool(x_h)
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