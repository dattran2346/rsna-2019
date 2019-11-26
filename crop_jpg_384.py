import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import glob
import os
import pandas as pd
from multiprocessing import Pool

f1 = './data/jpg/stage_2_test_images_prep'
f2 = './data/jpg/stage_2_test_images_crop'

list_f1 = glob.glob(os.path.join(f1, '*.jpg'))
list_f2 = glob.glob(os.path.join(f2, '*.jpg'))
list_f2 = [l.replace(f2, f1) for l in list_f2]
list_do = list(set(list_f1) - set(list_f2))
print(len(list_f1), len(list_f2), len(list_do))

def crop_image_from_gray(img,tol=50):
    if img.ndim ==2:
        mask = img>tol
        return img[np.ix_(mask.any(1),mask.any(0))]
    elif img.ndim==3:
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        mask = gray_img>tol

        check_shape = img[:,:,0][np.ix_(mask.any(1),mask.any(0))].shape[0]
        if (check_shape == 0):
            return img
        else:
            img1=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
            img2=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
            img3=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
            img = np.stack([img1,img2,img3],axis=-1)
        return img

def cropimg(path, sz=384):
    cropped_img = np.zeros(shape=(sz, sz, 3), dtype=np.uint8)
    img = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
    img = crop_image_from_gray(img)
    h, w, c = img.shape
    if not h < sz:
        img = img[:-(h-sz+2),:,:]
    if not w < sz:
        img = img[:,:-(w-sz+2),:]
    h, w, c = img.shape
    if h % 2 != 0:
        img = img[1:,:,:]
    if w % 2 != 0:
        img = img[:,1:,:]
    h, w, c = img.shape
    h_diff = (sz - h) // 2
    w_diff = (sz - w) // 2
    cropped_img[h_diff:-h_diff, w_diff:-w_diff ,:] = img

    cv2.imwrite(path.replace(f1, f2), cropped_img)

pool = Pool()
pool.map(cropimg, list_do)
pool.close()
pool.join()
