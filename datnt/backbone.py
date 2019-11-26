import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
import resnet
import pretrainedmodels
import warnings
from efficientnet_pytorch import EfficientNet
warnings.filterwarnings('ignore')

class ResNet18(nn.Module):
    def __init__(self, class_names):
        super(ResNet18, self).__init__()
        self.resnet = torchvision.models.resnet.resnet18(pretrained=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class ResNet34(nn.Module):
    def __init__(self, class_names):
        super(ResNet34, self).__init__()
        self.resnet = torchvision.models.resnet.resnet34(pretrained=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class ResNet50(nn.Module):
    def __init__(self, class_names):
        super(ResNet50, self).__init__()
        self.resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x    
    
class ResNet101(nn.Module):
    def __init__(self, class_names):
        super(ResNet101, self).__init__()
        self.resnet = torchvision.models.resnet.resnet101(pretrained=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x    
    
class ResNet152(nn.Module):
    def __init__(self, class_names):
        super(ResNet152, self).__init__()
        self.resnet = torchvision.models.resnet.resnet152(pretrained=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x    

class DilatedR50(nn.Module):
    def __init__(self, class_names):
        super(DilatedR50, self).__init__()
        self.resnet = resnet.resnet50(pretrained=True, dilated=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class DilatedR101(nn.Module):
    def __init__(self, class_names):
        super(DilatedR101, self).__init__()
        self.resnet = resnet.resnet101(pretrained=True, dilated=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class DilatedR152(nn.Module):
    def __init__(self, class_names):
        super(DilatedR152, self).__init__()
        self.resnet = resnet.resnet152(pretrained=True, dilated=True)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class WSL_8(nn.Module):
    def __init__(self, class_names):
        super(WSL_8, self).__init__()
        self.resnet = torch.hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl')
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x

class Xception(nn.Module):
    def __init__(self, class_names):
        super(Xception, self).__init__()
        self.base = pretrainedmodels.xception()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class DenseNet121(nn.Module):
    def __init__(self, class_names):
        super(DenseNet121, self).__init__()
        self.base = pretrainedmodels.densenet121()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class DenseNet169(nn.Module):
    def __init__(self, class_names):
        super(DenseNet169, self).__init__()
        self.base = pretrainedmodels.densenet169()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class DenseNet201(nn.Module):
    def __init__(self, class_names):
        super(DenseNet201, self).__init__()
        self.base = pretrainedmodels.densenet201()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x        

class DPN68(nn.Module):
    def __init__(self, class_names):
        super(DPN68, self).__init__()
        self.base = pretrainedmodels.dpn68()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(832, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x    

class EfficientNetB0(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB0, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1280, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB1(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB1, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b1')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1280, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB2(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB2, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b2')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1408, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB3(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB3, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b3')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1536, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB4(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB4, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b4')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(1792, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x 

class EfficientNetB5(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB5, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b5')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(2048, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x        

class EfficientNetB6(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB6, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b6')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(2304, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x  

class EfficientNetB7(nn.Module):
    def __init__(self, class_names):
        super(EfficientNetB7, self).__init__()
        self.base = EfficientNet.from_pretrained('efficientnet-b7')
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(2560, len(class_names))
    
    def forward(self, x):
        x = self.base.extract_features(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x         

class SEResNeXT50(nn.Module):
    def __init__(self, class_names):
        super(SEResNeXT50, self).__init__()
        self.base = pretrainedmodels.se_resnext50_32x4d()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x

class SEResNeXT101(nn.Module):
    def __init__(self, class_names):
        super(SEResNeXT101, self).__init__()
        self.base = pretrainedmodels.se_resnext101_32x4d()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x        

class InceptionV4(nn.Module):
    def __init__(self, class_names):
        super(InceptionV4, self).__init__()
        self.base = pretrainedmodels.inceptionv4()
        self.backbone = nn.Sequential(*list(self.base.children())[:-1])
        self.head = nn.Linear(self.base.last_linear.in_features, len(class_names))
    
    def forward(self, x):
        x = self.backbone(x)
        x = F.adaptive_avg_pool2d(x, 1).squeeze()
        x = self.head(x)
        return x  

class ResNet101AUX(nn.Module):
    def __init__(self, class_names):
        super(ResNet101AUX, self).__init__()
        self.resnet = torchvision.models.resnet.resnet101(pretrained=True)
        self.outclass = len(class_names)
        self.layer3 = nn.Sequential(*list(self.resnet.children())[:-3])
        self.layer4 = nn.Sequential(*list(self.resnet.children())[-3:-1])
        self.layer3aux = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1),
                                       nn.BatchNorm2d(2048),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3),
                                       nn.BatchNorm2d(1024),
                                       nn.ReLU())
        self.main_head = nn.Linear(2048, self.outclass)
        self.aux_head = nn.Linear(1024, self.outclass)

    def forward(self, x):
        l3o = self.layer3(x)
        l3o2 = self.layer3aux(l3o)
        aux = F.adaptive_avg_pool2d(l3o2, 1).squeeze()
        aux = self.aux_head(aux)

        l4o = self.layer4(l3o)
        main = F.adaptive_avg_pool2d(l4o, 1).squeeze()
        main = self.main_head(main)

        return (main, aux)

class ResNet50AUX(nn.Module):
    def __init__(self, class_names):
        super(ResNet50AUX, self).__init__()
        self.resnet = torchvision.models.resnet.resnet50(pretrained=True)
        self.outclass = len(class_names)
        self.layer3 = nn.Sequential(*list(self.resnet.children())[:-3])
        self.layer4 = nn.Sequential(*list(self.resnet.children())[-3:-1])
        self.layer3aux = nn.Sequential(nn.Conv2d(in_channels=1024, out_channels=2048, kernel_size=1),
                                       nn.BatchNorm2d(2048),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=2048, out_channels=1024, kernel_size=3),
                                       nn.BatchNorm2d(1024),
                                       nn.ReLU())
        self.main_head = nn.Linear(2048, self.outclass)
        self.aux_head = nn.Linear(1024, self.outclass)

    def forward(self, x):
        l3o = self.layer3(x)
        l3o2 = self.layer3aux(l3o)
        aux = F.adaptive_avg_pool2d(l3o2, 1).squeeze()
        aux = self.aux_head(aux)

        l4o = self.layer4(l3o)
        main = F.adaptive_avg_pool2d(l4o, 1).squeeze()
        main = self.main_head(main)

        return (main, aux)              

class ResNet34AUX(nn.Module):
    def __init__(self, class_names):
        super(ResNet34AUX, self).__init__()
        self.resnet = torchvision.models.resnet.resnet34(pretrained=True)
        self.outclass = len(class_names)
        self.layer3 = nn.Sequential(*list(self.resnet.children())[:-3])
        self.layer4 = nn.Sequential(*list(self.resnet.children())[-3:-1])
        self.layer3aux = nn.Sequential(nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
                                       nn.BatchNorm2d(512),
                                       nn.ReLU(),
                                       nn.Conv2d(in_channels=512, out_channels=256, kernel_size=3),
                                       nn.BatchNorm2d(256),
                                       nn.ReLU())
        self.main_head = nn.Linear(512, self.outclass)
        self.aux_head = nn.Linear(256, self.outclass)

    def forward(self, x):
        l3o = self.layer3(x)
        l3o2 = self.layer3aux(l3o)
        aux = F.adaptive_avg_pool2d(l3o2, 1).squeeze()
        aux = self.aux_head(aux)

        l4o = self.layer4(l3o)
        main = F.adaptive_avg_pool2d(l4o, 1).squeeze()
        main = self.main_head(main)

        return (main, aux)   

class InceptionV3(nn.Module):
    def __init__(self, class_names):
        super(InceptionV3, self).__init__()
        self.resnet = torchvision.models.inception_v3(pretrained=True, aux_logits=False)
        self.outclass = len(class_names)
        self.outfeature = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(self.outfeature, self.outclass)

    def forward(self, x):
        x = self.resnet(x)

        return x        

def make_model(base_model_name, class_names):
    if base_model_name == 'resnet18':
        return ResNet18(class_names).cuda()
    elif base_model_name == 'resnet34':
        return ResNet34(class_names).cuda()
    elif base_model_name == 'resnet34_aux':
        return ResNet34AUX(class_names).cuda()
    elif base_model_name == 'resnet50':
        return ResNet50(class_names).cuda()
    elif base_model_name == 'resnet50_aux':
        return ResNet50AUX(class_names).cuda()
    elif base_model_name == 'resnet101':
        return ResNet101(class_names).cuda()
    elif base_model_name == 'resnet101_aux':
        return ResNet101AUX(class_names).cuda()
    elif base_model_name == 'resnet152':
        return ResNet152(class_names).cuda()
    elif base_model_name == 'dilated50':
        return DilatedR50(class_names).cuda()
    elif base_model_name == 'dilated101':
        return DilatedR101(class_names).cuda()
    elif base_model_name == 'dilated152':
        return DilatedR152(class_names).cuda()
    elif base_model_name == 'xception':
        return Xception(class_names).cuda()
    elif base_model_name == 'densenet121':
        return DenseNet121(class_names).cuda()
    elif base_model_name == 'densenet169':
        return DenseNet169(class_names).cuda()
    elif base_model_name == 'densenet201':
        return DenseNet201(class_names).cuda()
    elif base_model_name == 'dpn68':
        return DPN68(class_names).cuda()
    elif base_model_name == 'b0':
        return EfficientNetB0(class_names).cuda()
    elif base_model_name == 'b1':
        return EfficientNetB1(class_names).cuda()
    elif base_model_name == 'b2':
        return EfficientNetB2(class_names).cuda()
    elif base_model_name == 'b3':
        return EfficientNetB3(class_names).cuda()
    elif base_model_name == 'b4':
        return EfficientNetB4(class_names).cuda()
    elif base_model_name == 'b5':
        return EfficientNetB5(class_names).cuda()
    elif base_model_name == 'b6':
        return EfficientNetB6(class_names).cuda()
    elif base_model_name == 'b7':
        return EfficientNetB7(class_names).cuda()   
    elif base_model_name == 'seresnext50':
        return SEResNeXT50(class_names).cuda()        
    elif base_model_name == 'seresnext101':
        return SEResNeXT101(class_names).cuda()
    elif base_model_name == 'inceptionv3':
        return InceptionV3(class_names).cuda()   
    elif base_model_name == 'inceptionv4':
        return InceptionV4(class_names).cuda()   
    elif base_model_name == 'wsl8':
        return WSL_8(class_names).cuda()
    else:
        raise ValueError('!!! MODELNAME !!!')     