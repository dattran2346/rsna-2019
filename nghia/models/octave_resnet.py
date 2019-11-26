import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from .octconv import oct_resnet50
from .utils_module import _initialize_weights

class OctResNet50(nn.Module):
    def __init__(self, model_name, input_channels, num_classes, pretrained=False, subtype_head=False, cfg=None):
        super(OctResNet50, self).__init__()
        self.subtype_head = subtype_head
        self.backbone = oct_resnet50()

        if pretrained:
            package_directory = os.path.dirname(os.path.abspath(__file__))
            pretrained_path = os.path.join(package_directory, 'octconv/oct_resnet50_cosine.pth')
            self.backbone.load_state_dict(torch.load(pretrained_path))
            print(f"loaded pretrained at {pretrained_path} !")

        if input_channels == 6:
            old_conv_weight = self.backbone.conv1.weight 
            new_conv = nn.Conv2d(6, 64, 7, 2, 3, bias=False)
            with torch.no_grad():
                new_conv.weight = nn.Parameter(torch.stack([torch.mean(old_conv_weight, 1)] * input_channels, 1))
            self.backbone.conv1 = new_conv

        del self.backbone.fc
        
        if self.subtype_head:
            self.fc_any = nn.Sequential(
                nn.Linear(512, 512),
                nn.ReLU(),
                nn.Linear(512, 1)
            )
            self.fc = nn.Linear(2048, out_features=num_classes-1, bias=True)

            _initialize_weights(self.fc_any)

        else:
            self.fc = nn.Linear(2048, out_features=num_classes, bias=True)
        nn.init.zeros_(self.fc.bias)

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
        if self.subtype_head:
            return [x_h_4, x_h_3] 
        return [x_h_4]

    def forward(self, x):
        features = self.feature(x)
        x = features[0]
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.subtype_head:
            x_any = features[1]
            x_any = x_any.view(x_any.size(0), -1)
            x_any = self.fc_any(x_any)
            x = torch.cat((x_any, x), 1)
        return x

    # def forward(self, x):
    #     x = self.backbone.conv1(x)
    #     x = self.backbone.bn1(x)
    #     x = self.backbone.relu(x)
    #     x = self.backbone.maxpool(x)

    #     x_h, x_l = self.backbone.layer1(x)
    #     x_h, x_l = self.backbone.layer2((x_h,x_l))

    #     x_h, x_l = self.backbone.layer3((x_h,x_l))
    #     x = self.backbone.avgpool(x_h)
    #     feature_aux = x.view(x.size(0), -1)

    #     x_h, x_l = self.backbone.layer4((x_h,x_l))
    #     x = self.backbone.avgpool(x_h)
    #     feature = x.view(x.size(0), -1)

    #     x = self.fc(feature)

    #     if self.subtype_head:
    #         x_any = self.fc_any(feature_aux)
    #         x = torch.cat((x_any, x), 1)
    #         return x
    #     return x