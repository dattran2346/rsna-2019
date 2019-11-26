import os
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
import torch
from torch.utils.data import Dataset
import numpy as np
from albumentations import *
from albumentations.imgaug.transforms import *
from albumentations.pytorch import *
from PIL import Image
from utils import dcm2img

class RSNADataset(Dataset):
    def __init__(self, df, root_dir, transform, datatype='train', tta = False):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        assert datatype == 'train' or datatype == 'test'
        self.datatype = datatype
        self.tta = tta

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dcm_path = os.path.join(self.root_dir, self.df.loc[idx, 'image'] + '.dcm')
        img = dcm2img(dcm_path)
        
        if self.tta:
            img_flip = img.copy()
            img_flip = np.fliplr(img_flip)
            image = self.transform(image=img)
            image = image['image']
            image_flip = self.transform(image=img_flip)
            image_flip = image_flip['image']
            if self.datatype == 'train':
                labels = self.df.loc[idx, ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
                return image, image_flip, torch.FloatTensor(labels)
            else:
                return image, image_flip
        else:
            image = self.transform(image=img)
            image = image['image']
            if self.datatype == 'train':
                labels = self.df.loc[idx, ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
                return image, torch.FloatTensor(labels)
            else:
                return image

class RSNADataset12TTAs(Dataset):
    def __init__(self, df, root_dir, cfg, zoom_ratio = 1.15):
        self.df = df.reset_index(drop=True)
        self.root_dir = root_dir
        self.cfg = cfg
        self.zoom_ratio = zoom_ratio
        self.zoom = Resize(height = int(self.zoom_ratio*self.cfg.size), width = int(self.zoom_ratio*self.cfg.size), always_apply=True, p=1)
        self.transform =  Compose([Normalize(), ToTensor()])
        self.resize = Resize(height = self.cfg.size, width = self.cfg.size, always_apply=True, p=1)
        
        self.tl = Crop(x_min=0, y_min=0, x_max=self.cfg.size, y_max=self.cfg.size, always_apply=True, p=1)
        self.tr = Crop(x_min=int(self.zoom_ratio*self.cfg.size)-self.cfg.size, y_min=0, x_max=int(self.zoom_ratio*self.cfg.size), y_max=self.cfg.size, always_apply=True, p=1)
        self.bl = Crop(x_min=0, y_min=int(self.zoom_ratio*self.cfg.size)-self.cfg.size, x_max=self.cfg.size, y_max=int(self.zoom_ratio*self.cfg.size), always_apply=True, p=1)
        self.br = Crop(x_min=int(self.zoom_ratio*self.cfg.size)-self.cfg.size, y_min=int(self.zoom_ratio*self.cfg.size)-self.cfg.size, 
                       x_max=int(self.zoom_ratio*self.cfg.size), y_max=int(self.zoom_ratio*self.cfg.size), always_apply=True, p=1)
        self.center = CenterCrop(height = self.cfg.size, width = self.cfg.size, always_apply=True, p=1)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        dcm_path = os.path.join(self.root_dir, self.df.loc[idx, 'image'] + '.dcm')
        img = dcm2img(dcm_path)
        imgz = self.zoom(image=img)['image']
        img1 = self.resize(image=img)['image']
        img2 = self.tl(image=imgz)['image']
        img3 = self.tr(image=imgz)['image']
        img4 = self.bl(image=imgz)['image']
        img5 = self.br(image=imgz)['image']
        img6 = self.center(image=imgz)['image']
        img7 = np.fliplr(img1)
        img8 = np.fliplr(img2)
        img9 = np.fliplr(img3)
        img10 = np.fliplr(img4)
        img11 = np.fliplr(img5)
        img12 = np.fliplr(img6)

        images = []
        for image in [img1, img2, img3, img4, img5, img6, img7, img8, img9, img10, img11, img12]:
            images.append(self.transform(image=image)['image'])
        return images

class StackingDataset(Dataset):
    def __init__(self, x = None, y = None, datatype='train'):
        self.x = x
        self.y = y
        assert datatype == 'train' or datatype == 'test'
        self.datatype = datatype

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        if self.datatype == 'train':
            return torch.FloatTensor(self.x[idx,:]), torch.FloatTensor(self.y[idx,:])
        else:
            return torch.FloatTensor(self.x[idx,:])

def getTransforms(size):
    transforms_train = Compose([
        Resize(size, size),
        OneOf([
            ShiftScaleRotate(
                shift_limit=0.0625,
                scale_limit=0.1,
                rotate_limit=30,
                border_mode=cv2.BORDER_CONSTANT,
                value=0),
            GridDistortion(
                distort_limit=0.2,
                border_mode=cv2.BORDER_CONSTANT,
                value=0),
            OpticalDistortion(
                distort_limit=0.2,
                shift_limit=0.15,
                border_mode=cv2.BORDER_CONSTANT,
                value=0),
            NoOp()
        ]),
        RandomSizedCrop(
            min_max_height=(int(size * 0.75), size),
            height=size,
            width=size,
            p=0.25),
        OneOf([
            RandomBrightnessContrast(
                brightness_limit=0.4,
                contrast_limit=0.4),
            RandomGamma(gamma_limit=(50, 150)),
            IAASharpen(),
            IAAEmboss(),
            CLAHE(clip_limit=2),
            NoOp()
        ]),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            MedianBlur(blur_limit=3),
            Blur(blur_limit=3),
        ], p=0.15),
        OneOf([
            RGBShift(),
            HueSaturationValue(),
        ], p=0.1),
        HorizontalFlip(p=0.5),
        Normalize(),
        ToTensor()
    ])
    transforms_test = Compose([
        Resize(size, size),
        Normalize(),
        ToTensor()
    ])
    return transforms_train, transforms_test