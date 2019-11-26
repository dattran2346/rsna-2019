import os
import torch
from torch.utils.data import Dataset
import numpy as np
import torch
import random
import pandas as pd
from sklearn.metrics import log_loss
import torch.nn as nn
import torchvision
import torch.nn.functional as F

def weightLogLoss(ytrue, pred):
    weight_log_loss = 0.0
    for i in range(6):
        yti = ytrue[:,i]
        # ypi = np.clip(ypred[:,i], 1e-7, 1 - 1e-7)
        ypi = pred[:,i]
        score = log_loss(yti, ypi)
        if i == 0:
            score *= 2
        weight_log_loss += score
    weight_log_loss /= 6.0
    return weight_log_loss

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

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

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