import numpy as np
import torch
import os
import random
import pydicom
import pandas as pd
from sklearn.metrics import log_loss

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

def seed_everything(seed=1234):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

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
    weight_log_loss /= 7.0
    return weight_log_loss

def get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

def get_windowing(data):
    dicom_fields = [data[('0028','1050')].value, #window center
                    data[('0028','1051')].value, #window width
                    data[('0028','1052')].value, #intercept
                    data[('0028','1053')].value] #slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def window_image(image, window_center,window_width):
    img = image.copy()
    img_min = window_center - window_width//2
    img_max = window_center + window_width//2
    img[img<img_min] = img_min
    img[img>img_max] = img_max
    if img.max() == img.min():
        return np.zeros(img.shape, dtype = np.uint8)
    else:
        img = 255*(img - img.min())/(img.max() - img.min())
        return img.astype(np.uint8)

def dcm2img(dcm_path):
    data = pydicom.read_file(dcm_path)
    image = data.pixel_array
    window_center, window_width, intercept, slope = get_windowing(data)
    image = (image*slope +intercept)
    image_c1 = window_image(image, window_center, window_width)
    image_c2 = window_image(image, 40, 80)
    image_c3 = window_image(image, 80, 200)
    return np.stack([image_c1, image_c2, image_c3], axis=-1)