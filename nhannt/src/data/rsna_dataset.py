import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)
from torch.utils.data import Dataset
import torch
import pandas as pd
import os
import numpy as np
from collections import OrderedDict
from albumentations import Compose, OneOf, HorizontalFlip, VerticalFlip, Resize, ShiftScaleRotate

from .sampler import dattran_slices_sampler, random_sorted_slices_sampler
from .utils import apply_window, pad_slices


WINDOWS = OrderedDict({
    'default': None,
    'brain': {'W': 80, 'L': 40},
    'subdural': {'W': 215, 'L': 75},
    'bony': {'W': 2800, 'L': 600},
    'tissue': {'W': 375, 'L': 40},
    'stroke1': {'W': 8, 'L': 32}})


class RSNAHemorrhageDS(Dataset):
    CLASSES = [
        "any",
        "intraparenchymal", "intraventricular",
        "subarachnoid", "subdural", "epidural"
    ]

    def __init__(self, cfg, mode="train",
                 rsna_windows=WINDOWS):
        super(RSNAHemorrhageDS, self).__init__()
        self.cfg = cfg
        self.mode = mode
        self.rsna_windows = rsna_windows
        self.train_root = cfg["DATA_DIR"] + cfg["TRAIN_FOLDER"]
        self.test_root = cfg["DATA_DIR"] + cfg["TEST_FOLDER"]

        train = pd.read_csv(cfg["DATA_DIR"] + cfg["TRAIN_CSV"])
        
        # extract study uids
        if self.cfg["SPLIT"] == "study":
            train_ids = np.load(
                cfg["DATA_DIR"] + "split_study/train_fold{}.npy".format(str(cfg["FOLD"])), allow_pickle=True)
            self.source = train[(train["StudyInstanceUID"].isin(train_ids))] \
                .reset_index(drop=True)
            valid_ids = np.load(
                cfg["DATA_DIR"] + "split_study/valid_fold{}.npy".format(str(cfg["FOLD"])), allow_pickle=True)
            if self.cfg["FILTER_NO_BRAIN"]:
                self.target = train[(train["StudyInstanceUID"].isin(
                    valid_ids))&(train["BrainPresence"]==True)].reset_index(drop=True)
            else:
                self.target = train[train["StudyInstanceUID"].isin(
                    valid_ids)].reset_index(drop=True)
        # extract patient ids
        elif self.cfg["SPLIT"] == "patient":
            train_ids = np.load(
                cfg["DATA_DIR"] + "split_patients/train_fold{}.npy".format(str(cfg["FOLD"])), allow_pickle=True)
            self.source = train[(train["PatientID"].isin(train_ids))] \
                .reset_index(drop=True)
            valid_ids = np.load(
                cfg["DATA_DIR"] + "split_patients/valid_fold{}.npy".format(str(cfg["FOLD"])), allow_pickle=True)
            if self.cfg["FILTER_NO_BRAIN"]:
                self.target = train[(train["PatientID"].isin(
                    valid_ids))&(train["BrainPresence"]==True)].reset_index(drop=True)
            else:
                self.target = train[train["PatientID"].isin(
                    valid_ids)].reset_index(drop=True)
              
        test = pd.read_csv(cfg["DATA_DIR"] + cfg["TEST_CSV"])
        self.test = test.reset_index(drop=True)

        if self.cfg["MODEL_NAME"].startswith("tf_efficient"):
            resize_mode = cv2.INTER_CUBIC
        elif "res" in cfg["MODEL_NAME"]:
            resize_mode = cv2.INTER_LINEAR
        
        self.resize = Resize(cfg["IMG_SIZE"], cfg["IMG_SIZE"],
                             interpolation=resize_mode, always_apply=True)
        self.train_transform = Compose([
            OneOf([
                HorizontalFlip(1.),
                VerticalFlip(1.)
            ], p=cfg["P_AUGMENT"]),
            ShiftScaleRotate(shift_limit=0.0234375, scale_limit=(-0.1, 0.),
                             rotate_limit=10, p=cfg["P_AUGMENT"],
                             interpolation=resize_mode),
        ])

        self.train_studies = self.source["StudyInstanceUID"].unique()
        self.valid_studies = self.target["StudyInstanceUID"].unique()
        self.test_studies = self.test["StudyInstanceUID"].unique()

    def _load_hu_img(self, file_path):
        hu_img = np.load(file_path, allow_pickle=True)

        if hu_img.shape[0] != self.cfg["IMG_SIZE"]:
            hu_img = self.resize(image=hu_img)["image"]

        if self.mode == "train":
            hu_img = self.train_transform(image=hu_img)["image"]

        hu_img = hu_img[np.newaxis, :]
        img_tensor = torch.from_numpy(hu_img).type('torch.FloatTensor')
        return img_tensor


class RSNAHemorrhageDS2d(RSNAHemorrhageDS):
    def __init__(self, cfg, mode="train",
                 rsna_windows=WINDOWS):
        super(RSNAHemorrhageDS2d, self).__init__(cfg, mode, rsna_windows) 

    def __len__(self):
        if self.mode == "train":
            return len(self.source)
        elif self.mode == "valid":
            return len(self.target)
        elif self.mode == "test":
            return len(self.test)

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
        img = self._load_hu_img(img_path)
        if self.mode == "train" or self.mode == "valid":
            return img, torch.FloatTensor([info["any"],
                                           info["intraparenchymal"], info["intraventricular"],
                                           info["subarachnoid"], info["subdural"], info["epidural"]])
        else:
            img_id = info["image"]
            return img, img_id


class RSNAHemorrhageDS3d(RSNAHemorrhageDS):
    def __init__(self, cfg, mode="train",
                 rsna_windows=WINDOWS):
        super(RSNAHemorrhageDS3d, self).__init__(cfg, mode, rsna_windows) 

    def __len__(self):
        if self.mode == "train":
            return len(self.train_studies)
        elif self.mode == "valid":
            return len(self.valid_studies)
        elif self.mode == "test":
            return len(self.test_studies)

    def _load_study(self, study_df):
        img_names = study_df["image"].values
        if not self.mode == "test":
            labels = study_df[self.CLASSES].values        

        if self.mode == "train":
            nslices = self.cfg["NUM_SLICES"]
            l = len(img_names)            
            if np.random.uniform() < self.cfg["P_AUGMENT"]:
                idx = dattran_slices_sampler(nslices, l)
            else:
                idx = random_sorted_slices_sampler(nslices, l)
            img_names = img_names[idx]
            labels = labels[idx]

        if self.mode == "train" or self.mode == "valid":
            data_root = self.train_root
        elif self.mode == "test":
            data_root = self.test_root

        img_paths = [os.path.join(data_root, img_name + ".npy")
                     for img_name in img_names]
        imgs = [self._load_hu_img(img_path) for img_path in img_paths]
        imgs = torch.stack(imgs)

        if self.mode == "train" or self.mode == "valid": 
            return imgs, torch.from_numpy(
                labels).type('torch.FloatTensor')

        elif self.mode == "test":
            return imgs, img_names

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
        data_bunch = self._load_study(study_df)
        return data_bunch
