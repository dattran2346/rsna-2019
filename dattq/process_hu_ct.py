import cv2
import numpy as np
import matplotlib.pyplot as plt
import pydicom
from tqdm import tqdm_notebook
import glob
import os
import pandas as pd
from multiprocessing import Pool
from pathlib import Path
from joblib import Parallel, delayed
import argparse


parser = argparse.ArgumentParser(
            description='Preprocess data ')

parser.add_argument('--data-dir', default='/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/', type=str, help='Base data dir')
parser.add_argument('--output-dir', default='/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/hu_origin/', type=str, help='Output dir')
args = parser.parse_args()

input_dir = Path(args.data_dir)
output_dir = Path(args.output_dir)

train_folder = 'stage_1_train_images'
test_folder = 'stage_1_test_images'

train_img_paths = (input_dir/train_folder).glob('*dcm')
test_img_paths = (input_dir/test_folder).glob('*dcm')

train_output_dir = output_dir/train_folder
test_output_dir = output_dir/test_folder
train_output_dir.mkdir(parents=True, exist_ok=True)
test_output_dir.mkdir(parents=True, exist_ok=True)

def preprocess(img_path, output_dir):
    image_name = img_path.stem

    # read dicom file
    r = pydicom.read_file(img_path.as_posix())

    # convert to hounsfield unit
    img = (r.pixel_array * r.RescaleSlope) + r.RescaleIntercept # float64
    img = img.astype(np.int16) # need int16 to store only

    np.save(output_dir/f'{image_name}.npy', img, allow_pickle=True)
    return

def catch_wrapper(img_path, output_dir):
    try:
        preprocess(img_path, output_dir)
    except Exception as e:
        print(e, img_path.stem)

print(f'======================== Convert test dicom to HU ========================')
Parallel(n_jobs=os.cpu_count(), verbose=1)(delayed(catch_wrapper)(f, test_output_dir) for f in test_img_paths)


print(f'======================== Convert train dicom to HU ========================')
Parallel(n_jobs=os.cpu_count(), verbose=1)(delayed(catch_wrapper)(f, train_output_dir) for f in train_img_paths)
