import os

import PIL
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from albumentations import *
from albumentations.pytorch import ToTensor

from utils import *

def apply_window(window, hu_img):
    l, w = window["L"], window["W"]
    window_min = l - (w // 2)
    window_max = l + (w // 2)
    img = np.clip(hu_img, window_min, window_max)    
    img = 255 * ((img - window_min) / w)
    img = img.astype(np.uint8)
    return img


def calculate_pad(height, width, target_height, target_width):
    pad_height = target_height - height
    pad_width = target_width - width
    top = pad_height // 2
    bottom = pad_height - top
    left = pad_width // 2
    right = pad_width - left
    return top, bottom, left, right


def crop(img, tol=0):
    mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

class DebugSampler(Sampler):

    def __init__(self, mask):
        self.mask = mask

    def __iter__(self):
        return (self.indices[i] for i in torch.nonzero(self.mask))

    def __len__(self):
        return len(self.mask)

class EasySampler(Sampler):
    """Samples easy elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, hard_indices, easy_indices, ratio=0.2):
        self.hard_indices = hard_indices
        self.easy_indices = easy_indices
        self.ratio = ratio
        self.length = len(hard_indices) + int(len(hard_indices) * ratio)

    def __iter__(self):
        sampled_easy_indices = np.random.choice(self.easy_indices,
                                                size=int(
                                                    len(self.hard_indices) * self.ratio),
                                                replace=False)
        indices = np.append(self.hard_indices, sampled_easy_indices)
        return (indices[i] for i in torch.randperm(len(indices)))

    def __len__(self):
        return self.length

class RSNAHemorrhageDS(Dataset):
    CLASSES = [
        "any", 
        "intraparenchymal", "intraventricular", 
        "subarachnoid", "subdural", "epidural"
    ]
    def __init__(self, cfg, mode="train"):
        super(RSNAHemorrhageDS, self).__init__()
        self.mode = mode
        self.cfg = cfg
        self.rsna_windows = {
                                'default': None,
                                'brain': {'W': 80, 'L': 40 },
                                'subdural': {'W': 215, 'L': 75},
                                'bony': { 'W': 2800, 'L': 600 },
                                'tissue': {'W': 375, 'L': 40},
                                'stroke1': {'W': 8, 'L': 32}
                                # 'stroke2': {'W': 40, 'L': 40}
                            }
        self.train_root = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN_IMAGES)
        self.test_root = os.path.join(cfg.DIRS.DATA, cfg.DIRS.TEST_IMAGES)

        train = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TRAIN_META))
        # train = train[train["BrainPresence"]==True]
        train_ids = np.load(os.path.join(cfg.DIRS.DATA,"split_study", f"train_fold{cfg.TRAIN.FOLD}.npy"),
                            allow_pickle=True)
        self.source = train[train["StudyInstanceUID"].isin(train_ids)].reset_index(drop=True)

        # duplicate with site
        if cfg.DATA.INP_CHANNEL == 3 and not cfg.DATA.SITE_ON_THE_FLY:
            if not cfg.DATA.ONE_SITE:
                site = np.append([0] * len(self.source), [1] * len(self.source))
                self.source = pd.concat([self.source, self.source], 0)
            else:
                site = [0] * len(self.source)
            self.source = self.source.reset_index(drop=True)
            self.source["site"] = site

        valid_ids = np.load(os.path.join(cfg.DIRS.DATA, "split_study", f"valid_fold{cfg.TRAIN.FOLD}.npy"), 
                            allow_pickle=True)
        
        # valid = train[train["BrainPresence"]==True]
        valid = train.copy()
        self.target = valid[valid["StudyInstanceUID"].isin(valid_ids)].reset_index(drop=True)

        test = pd.read_csv(os.path.join(cfg.DIRS.DATA, cfg.DIRS.TEST_META))
        self.test = test.reset_index(drop=True)
        # test_removed_ids = np.load(os.path.join(cfg.DIRS.DATA, "test_removed_ids.npy"), 
        #                            allow_pickle=True)
        # self.test = test[~test["image"].isin(test_removed_ids)].reset_index(drop=True)

        IMG_SIZE = cfg.DATA.IMG_SIZE

        self.train_transform = Compose([
            RandomResizedCrop(IMG_SIZE, IMG_SIZE, 
                              scale=(0.75, 1.0), ratio=(1.0, 1.0), 
                            always_apply=True) if IMG_SIZE < 512 else Resize(512, 512, always_apply=True),
            
            OneOf([
                HorizontalFlip(always_apply=True),
                VerticalFlip(always_apply=True)
            ], p=1.),

            OneOf([
                ShiftScaleRotate(shift_limit=0.0234375, scale_limit=(-0.1, 0.), 
                    rotate_limit=10, always_apply=True),
                MedianBlur(always_apply=True)
            ], p=1.),

        ], p=cfg.DATA.PAUGMENT)

        self.val_transform = Compose([
            Resize(IMG_SIZE, IMG_SIZE, always_apply=True)
        ])

    def __len__(self):
        if self.mode == "train":
            return len(self.source)
        elif self.mode == "valid":
            return len(self.target)
        elif self.mode == "test":
            return len(self.test)

    def _load_img(self, file_path, info=None, site=0):
        hu_img = np.load(file_path, allow_pickle=True)
        
        current_windows = self.rsna_windows
        current_windows["default"] = {'W': info["WindowWidth"], 
                                      'L': info["WindowCenter"]}

        if self.cfg.DATA.INP_CHANNEL == 6:
            custom_windows = current_windows
        if self.cfg.DATA.INP_CHANNEL == 3:
            window_types = self.cfg.DATA.RSNA1 if site == 0 else self.cfg.DATA.RSNA2
            custom_windows = {k:current_windows[k] for k in window_types if k in current_windows}

        img_tensor = [apply_window(window, hu_img) for window_name, window in custom_windows.items()]

        # per channel augment
        if self.cfg.DATA.AUGMENT_PER_CHANNEL:
            if self.mode == "train":
                img_tensor = [self.train_transform(image=img)["image"] for img in img_tensor]
            else:
                img_tensor = [self.val_transform(image=img)["image"] for img in img_tensor]
            img_tensor = [torch.from_numpy(img[np.newaxis, :]) for img in img_tensor]
            img_tensor = torch.cat(img_tensor).type('torch.FloatTensor')
            img_tensor = img_tensor / 255.
        # all channel
        else:
            transform = Compose([
                self.train_transform if self.mode == "train" else self.val_transform,
                Normalize(),
                ToTensor()
            ])

            img_tensor = np.stack(img_tensor, axis=2)
            img_tensor = transform(image=img_tensor)['image']

        return img_tensor        

    def __getitem__(self, idx):
        if self.mode == "train":
            info = self.source.loc[idx]
        elif self.mode == "valid":
            info = self.target.loc[idx]
        elif self.mode == "test":
            info = self.test.loc[idx]
        
        if self.mode != "test":
            data_root = self.train_root
        else:
            data_root = self.test_root 
        
        img_path = os.path.join(data_root, info["image"] + ".npy")
        
        if self.mode == "train":
            if self.cfg.DATA.SITE_ON_THE_FLY:
                site_info = 1 if np.random.uniform() > 0.5 else 0
            else:
                site_info = info['site'] 
            img = self._load_img(img_path, info, site_info)
        else:
            if self.cfg.DATA.INP_CHANNEL == 3 and not self.cfg.DATA.ONE_SITE:
                img1 = self._load_img(img_path, info, 0)
                img2 = self._load_img(img_path, info, 1)
                img = [img1, img2]
            else:
                img = self._load_img(img_path, info)

        
        infos = torch.FloatTensor([info["any"]]) if self.cfg.TRAIN.NUM_CLASSES  == 1 else \
                torch.FloatTensor([info["any"], 
                                           info["intraparenchymal"],	info["intraventricular"], 
                                           info["subarachnoid"], info["subdural"], info["epidural"]])
        if self.mode == "train":
            return img, infos
        elif self.mode == "valid":
            return img, infos
        else:
            img_id = info["image"]
            return img, img_id

class RSNAHemorrhageDS_RNN(RSNAHemorrhageDS):

    def __init__(self, cfg, mode="train"):
        super(RSNAHemorrhageDS_RNN, self).__init__(cfg, mode)
        if cfg.TRAIN.CRNN:
            self.train_studies = self.source["StudyInstanceUID"].unique()
            self.valid_studies = self.target["StudyInstanceUID"].unique()
            self.test_studies = self.test["StudyInstanceUID"].unique()
        if cfg.DATA.JPG != "":
            self.transform = MegaAugmenter(cfg.DATA.IMG_SIZE) if mode == "train" else ValAugmenter(cfg.DATA.IMG_SIZE)

    def __len__(self):
        if self.mode == "train":
            return len(self.train_studies)
        elif self.mode == "valid":
            return len(self.valid_studies)
        elif self.mode == "test":
            return len(self.test_studies)

    def _load_study(self, study_df):
        img_names = study_df["image"].values
        infos = [{'WindowWidth': ww, 
                  'WindowCenter': wc}
                  for ww, wc in zip(study_df["WindowWidth"].values, study_df["WindowCenter"].values)]


        if self.cfg.TRAIN.NUM_CLASSES  == 1:
            self.CLASSES = ["any"]
            
        if self.mode != "test":
            labels = study_df[self.CLASSES].values
            data_root = self.train_root
        else:
            data_root = self.test_root 

        
        if self.cfg.DATA.JPG:
            img_paths = [os.path.join(self.cfg.DIRS.JPGS, img_name + ".jpg") for img_name in img_names]
            imgs = [self._load_img_jpg(img_path) for img_path in img_paths]
        else:
            img_paths = [os.path.join(data_root, img_name + ".npy") for img_name in img_names]
            imgs = [self._load_img(img_path, info, site=0) for img_path, info in zip(img_paths, infos)]


        n_img = len(imgs)
        if self.mode == "train" and self.cfg.DATA.N_SLICES > 0:
            if self.cfg.DATA.N_SLICES > n_img:
                imgs = imgs + [torch.zeros(self.cfg.DATA.INP_CHANNEL, self.cfg.DATA.IMG_SIZE, self.cfg.DATA.IMG_SIZE)] * (self.cfg.DATA.N_SLICES - n_img)
                labels = np.append(labels, np.zeros((self.cfg.DATA.N_SLICES - n_img, len(self.CLASSES)), dtype=np.int64), axis=0)
            else:
                start_idx = np.random.randint(0, n_img - self.cfg.DATA.N_SLICES + 1)
                end_idx = start_idx + self.cfg.DATA.N_SLICES
                imgs = imgs[start_idx:end_idx]
                labels = labels[start_idx:end_idx]

        imgs = torch.stack(imgs)

        if self.mode == "train": 
            return imgs, torch.from_numpy(labels).type('torch.FloatTensor')
        elif self.mode == "valid":
            return imgs, torch.from_numpy(labels).type('torch.FloatTensor'), list(img_names)
        elif self.mode == "test":
            return imgs, list(img_names)

    def _load_img_jpg(self, filepath):
        img = PIL.Image.open(filepath).convert("RGB")
        img = np.array(img) 
        return self.transform(img)

    def __getitem__(self, idx):
        if self.mode == "train":
            study_name = self.train_studies[idx]
            study_df = self.source[self.source["StudyInstanceUID"] == study_name]
        elif self.mode == "valid":
            study_name = self.valid_studies[idx]
            study_df = self.target[self.target["StudyInstanceUID"] == study_name]
        elif self.mode == "test":
            study_name = self.test_studies[idx]
            study_df = self.test[self.test["StudyInstanceUID"] == study_name]
        data_bunch = self._load_study(study_df.reset_index(drop=True))
        return data_bunch

"""
    JPG DatNgo Augmenter
"""

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