import ast
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class ImetDataset(Dataset):
    """
    Imet dataset.
    """
    def __init__(self, data_dir, csv_file, mode, transform):
        self.data_dir = data_dir
        self.mode = mode
        self.transform = transform

        labels_df = pd.read_csv(os.path.join(self.data_dir, "labels.csv"))
        self.classes = {k: v for k, v in zip(labels_df['attribute_id'].values, 
            labels_df['attribute_name'].values)}
        
        self.df = pd.read_csv(os.path.join(self.data_dir, csv_file))
        self.imgs = self.df["id"].values
        self.id_to_img_map = {k: v for k, v in enumerate(self.imgs)}
        
        if not self.mode == 'test':
            self.labels = self.df["attribute_ids"].values
                
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, index):
        img_file = self.imgs[index]
        if self.mode == 'test':
            img_path = os.path.join(self.data_dir, "test", img_file + '.png')
        else:
            img_path = os.path.join(self.data_dir, "train", img_file + '.png') 
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        if self.mode == 'test':
            return img, img_file
        else:
            label = self.labels[index].split(' ')
            label = [int(l) for l in label]
            one_hot_label = torch.zeros(len(self.classes))
            one_hot_label = one_hot_label.scatter_(0, torch.LongTensor(label), 1.0)

            # print(img_file, label)

            if self.mode == 'val':
                return img, one_hot_label     
            
            return img, one_hot_label[:398], one_hot_label[398:]