import cv2
import numpy as np
import os
import pandas as pd
from torch.utils.data import Dataset, Sampler
from PIL import Image
from skimage.transform import resize
from skimage.color import gray2rgb
from skimage import exposure
from skimage.filters import median
from skimage.morphology import disk
from skimage import img_as_float
from matplotlib import pyplot as plt
from torch import Tensor
from torchvision.transforms import ToTensor
from torchvision import transforms
import torch
from PIL import Image

class RSNADataset(Dataset):
    def __init__(self, dataset_csv_file, class_names, source_image_dir, augmenter=None):
        self.df = pd.read_csv(dataset_csv_file)
        self.class_names = class_names
        self.source_image_dir = source_image_dir
        self.augmenter = augmenter
        self.x_path, self.y = self.df["image"].values, self.df[self.class_names].values

    def __len__(self):
        return len(self.x_path)

    def __getitem__(self, idx):
        image = self.load_image(self.x_path[idx]+'.jpg')
        image = self.transform_image(image)
        y = Tensor(self.y[idx])
        return image, y


    def load_image(self, image_file):

        """
        Load and crop images using template matching technique.
        Input:  Image file
        Output: Cropped image that matches to a given template. Due to the CheXpert dataset provides both frontal and lateral images, so two templates will be used.
        """

        # Read the image and take its name.
        image_path = os.path.join(self.source_image_dir, image_file)
        img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
        return img
          

    def transform_image(self, image):
        """
        Run data augmentation on a batch of images. Then, perform data normalization with mean and std. of ImageNet.
        """
        if self.augmenter is not None:
            image = self.augmenter(image)
        return image