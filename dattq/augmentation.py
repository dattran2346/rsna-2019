from torchvision import transforms
import torch
import cv2
import random
import numpy as np
import torchvision.transforms.functional as TF
import random


import albumentations as A
from albumentations.pytorch import ToTensor

from albumentations.augmentations import functional as F
from albumentations.core.transforms_interface import DualTransform

BRAIN_MEAN, BRAIN_STD = 41.25654302, 80.03385203
SUBDURAL_MEAN, SUBDURAL_STD = 35.64595268, 72.07324594
BONY_MEAN, BONY_STD = 29.72877898, 45.39197281
TISSUE_MEAN, TISSUE_STD = 46.52425779, 77.81010204
STROKE1_MEAN, STROKE1_STD = 38.12247465, 81.28239066
STROKE2_MEAN, STROKE2_STD = 44.59075768, 93.56715624

def get_stat(mix_window):
    if mix_window == 1:
        # use imagenet stat
        MEAN = np.array([0.48560741861744905, 0.49941626449353244, 0.43237713785804116])
        STD = np.array([0.2321024260764962, 0.22770540015765814, 0.2665100547329813])
    elif mix_window == 3:
        MEAN = np.array([BRAIN_MEAN, SUBDURAL_MEAN, BONY_MEAN]) / 255
        STD = np.array([BRAIN_STD, SUBDURAL_STD, BONY_STD]) / 255
    elif mix_window == 6:
        MEAN = np.array([BRAIN_MEAN, SUBDURAL_MEAN, BONY_MEAN, TISSUE_MEAN, STROKE1_MEAN, STROKE2_MEAN]) / 255
        STD = np.array([BRAIN_STD, SUBDURAL_STD, BONY_STD, TISSUE_STD, STROKE1_STD, STROKE2_STD]) / 255

    return MEAN, STD

class HarderAugmenter:
    def __init__(self, args):
        MEAN, STD = get_stat(args.mix_window)
        self.transform = A.Compose([
            A.RandomResizedCrop(args.image_size, args.image_size, interpolation=cv2.INTER_LINEAR, scale=(0.8, 1)),
            # A.Resize(args.image_size, args.image_size),
            A.HorizontalFlip(),
            A.VerticalFlip(), # add this -> freebie in tta stage

            ################################
            # Test augmentation
            ################################
            A.OneOf([ A.ShiftScaleRotate(
                        shift_limit=0.0625,
                        scale_limit=0.1,
                        rotate_limit=30,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0),
                      A.GridDistortion(
                        distort_limit=0.2,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0),
                      A.OpticalDistortion(
                        distort_limit=0.2,
                        shift_limit=0.15,
                        border_mode=cv2.BORDER_CONSTANT,
                        value=0),
                      A.NoOp()
            ]),

            A.OneOf([
                A.IAAAdditiveGaussianNoise(),
                A.GaussNoise(),
                A.MedianBlur(blur_limit=3),
                A.Blur(blur_limit=3),
            ], p=0.15),

            A.Normalize(MEAN, STD),
            ToTensor()
        ])

    def __call__(self, x):
        return self.transform(image=x)['image']


class ValAugmenter:

    def __init__(self, args):
        MEAN, STD = get_stat(args.mix_window)
        self.transform = A.Compose([
            # A.Resize(args.image_size, args.image_size),
            A.Normalize(MEAN, STD),
            ToTensor()
        ])

    def __call__(self, x):
        return self.transform(image=x)['image']

class TestAugmenter:
    def __init__(self, args):
        MEAN, STD = get_stat(args.mix_window)
        self.transform = A.Compose([
            A.Resize(args.image_size, args.image_size),
            A.Normalize(MEAN, STD),
            ToTensor()
        ])

    def __call__(self, x):
        return self.transform(image=x)['image']
