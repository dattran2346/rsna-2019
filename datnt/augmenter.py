from torchvision import transforms
import torch
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import random
import numpy as np
import torchvision.transforms.functional as TF
import random

from albumentations import Compose, HorizontalFlip, VerticalFlip, Normalize, Resize, \
            RandomResizedCrop, MotionBlur, RandomBrightness, RandomContrast, OneOf, RandomGamma, \
            CLAHE, Blur, CoarseDropout, GaussNoise, HueSaturationValue, ShiftScaleRotate, Transpose, \
            ElasticTransform, IAAAdditiveGaussianNoise, IAASharpen, RandomBrightnessContrast, \
            GridDistortion, RandomResizedCrop, OpticalDistortion, NoOp, MedianBlur
from albumentations.pytorch import ToTensor

class HarderAugmenter:
    def __init__(self, target_size):
        transformation = []
        transformation += [Resize(target_size, target_size)]
        transformation += [RandomResizedCrop(target_size, target_size, scale=(0.8, 1.0)),
                           ShiftScaleRotate(),
                           ElasticTransform(approximate=True),
                           GaussNoise(),
                           ]
        transformation += [Normalize(),
                           ToTensor()]
        print(transformation)                           
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class MegaAugmenter:
    def __init__(self, target_size):
        transformation = []
        transformation += [Resize(target_size, target_size)]
        transformation += [RandomResizedCrop(target_size, target_size, scale=(0.8, 1.0)),
                           OneOf([
                               ShiftScaleRotate(),
                               GridDistortion(),
                               OpticalDistortion(),
                               ElasticTransform(approximate=True)], p=0.3),
                           OneOf([
                               IAAAdditiveGaussianNoise(),
                               GaussNoise(),
                               MedianBlur(blur_limit=3),
                               Blur(blur_limit=3)], p=0.3)]
        transformation += [Normalize(),
                           ToTensor()]
        print(transformation)                           
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class ValAugmenter:
    def __init__(self, target_size):
        transformation = []
        transformation += [Resize(target_size, target_size)]
        transformation += [Normalize(),
                           ToTensor()]
        self.transform = Compose(transformation)

    def __call__(self, x):
        return self.transform(image=x)['image']

class FiveCropAugmenter:
    def __init__(self, target_size):
        transformation = []
        transformation += [transforms.ToPILImage(),
                           transforms.Resize((target_size, target_size)),
                           transforms.FiveCrop(int(target_size*0.9)),
                           transforms.Lambda(lambda crops: torch.stack([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))(transforms.ToTensor()(transforms.Resize((target_size, target_size))(crop))) for crop in crops]))]

        self.transforms = transforms.Compose(transformation)

    def __call__(self, x):
        return self.transforms(x)
