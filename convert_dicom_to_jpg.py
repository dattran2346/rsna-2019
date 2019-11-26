import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
import glob
import os
import pandas as pd
from multiprocessing import Pool

f1 = './data/dicom/stage_2_test_images'
f2 = './data/jpg/stage_2_test_images_prep'

list_dcm = glob.glob(os.path.join(f1, '*.dcm'))
list_png = glob.glob(os.path.join(f2, '*.jpg'))
list_png = [l.replace('.jpg', '.dcm').replace(f2, f1) for l in list_png]
list_do = set(list_dcm) - set(list_png)
print(len(list_dcm), len(list_png), len(list_do))

def preprocess(path):
    # params
    window_min = 0
    window_max = 120

    # read dicom file
    r = pydicom.read_file(path)

    # convert to hounsfield unit
    img = (r.pixel_array * r.RescaleSlope) + r.RescaleIntercept

    # apply brain window
    img = np.clip(img, window_min, window_max)
    img = img * 255 / window_max
    img = img.astype(np.uint8)

    # create binary mask
    binary_mask = (img > 0).astype(np.uint8)

    # keep only biggest connected component in mask
    ret, labels = cv2.connectedComponents(binary_mask)
    biggest_label = 0
    biggest_area = 0
    for label in range(1,ret):
        mask = np.array(labels, dtype=np.uint8)
        mask[labels == label] = 1
        mask[labels != label] = 0
        area = np.sum(mask)
        if area > biggest_area:
            biggest_area = area
            biggest_label = label

    remained_mask = np.array(labels, dtype=np.uint8)
    remained_mask[labels == biggest_label] = 1
    remained_mask[labels != biggest_label] = 0

    # apply mask to windowed image
    preprocessed_img = remained_mask * img

    cv2.imwrite(path.replace(f1, f2).replace('dcm', 'jpg'), preprocessed_img)

    return

pool = Pool()
pool.map(preprocess, list_do)
pool.close()
pool.join()
