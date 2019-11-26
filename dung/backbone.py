import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import pretrainedmodels
import warnings
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings('ignore')

class EfficientNetB0(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB0, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB1(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB1, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b1')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1280, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB2(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB2, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b2')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1408, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB3(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB3, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1536, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB4(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB4, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b4')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1792, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB5(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB5, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b5')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class EfficientNetB6(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB6, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b6')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(2304, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class EfficientNetB7(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetB7, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b7')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(2560, num_classes)
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class ResNet18(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        self.outclass = num_classes
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, num_classes):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet.resnet34(pretrained=True)
        self.outclass = num_classes
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.outclass = num_classes
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class ResNet101(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101, self).__init__()
        self.resnet = torchvision.models.resnet.resnet101(pretrained=True)
        self.outclass = num_classes
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class ResNet152(nn.Module):
    def __init__(self, num_classes):
        super(ResNet152, self).__init__()
        self.resnet = torchvision.models.resnet.resnet152(pretrained=True)
        self.outclass = num_classes
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class Xception(nn.Module):
    def __init__(self, num_classes):
        super(Xception, self).__init__()
        self.base = pretrainedmodels.xception()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet121, self).__init__()
        self.base = pretrainedmodels.densenet121()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class DenseNet169(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet169, self).__init__()
        self.base = pretrainedmodels.densenet169()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class SEResNeXT50(nn.Module):
    def __init__(self, num_classes):
        super(SEResNeXT50, self).__init__()
        self.base = pretrainedmodels.__dict__['se_resnext50_32x4d'](pretrained='imagenet')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class SEResNeXT101(nn.Module):
    def __init__(self, num_classes):
        super(SEResNeXT101, self).__init__()
        self.base = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained='imagenet')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class InceptionV4(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV4, self).__init__()
        self.base = pretrainedmodels.inceptionv4()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x  

class InceptionV3(nn.Module):
    def __init__(self, num_classes):
        super(InceptionV3, self).__init__()
        self.resnet = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        self.outclass = num_classes
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class Resnext101_32x8d_WSL(nn.Module):
    def __init__(self, num_classes):
        super(Resnext101_32x8d_WSL, self).__init__()
        self.base = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.fc.in_features, num_classes)
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

def make_model(model_name, num_classes):
    if model_name == 'EfficientnetB2':
        return EfficientNetB2(num_classes)
    elif model_name == 'EfficientnetB3':
        return EfficientNetB3(num_classes)
    elif model_name == 'EfficientnetB4':
        # return EfficientNetB4(num_classes)
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=num_classes)
        return model
    elif model_name == 'EfficientnetB5':
        return EfficientNetB5(num_classes)
    elif model_name == 'Resnext101_32x8d_WSL':
        return Resnext101_32x8d_WSL(num_classes)
    elif model_name == 'SE-ResNeXt101_32x4d':
        # return SEResNeXT101(num_classes)
        model = pretrainedmodels.__dict__['se_resnext101_32x4d'](pretrained='imagenet')
        num_ftrs = model.last_linear.in_features
        model.last_linear = nn.Linear(in_features=num_ftrs, out_features=num_classes,bias=True)
        return model
    elif model_name == 'InceptionV4':
        return InceptionV4(num_classes)
    else:
        raise ValueError('!!! MODELNAME !!!')

def StackingModel1(input_dim):
    model = nn.Sequential(
        nn.Linear(input_dim, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 1024),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(1024, 6),
    )
    return model