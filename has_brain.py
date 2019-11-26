import pydicom
from joblib import Parallel, delayed
import shutil
from pathlib import Path
import os
import numpy as np
import cv2
import pandas as pd
import pydicom

test_dir = Path('data/dicom/stage_2_test_images')
test_img_paths = test_dir.glob('*.dcm')
#test_img_paths = list(test_img_paths)[:1000]

window_min = 0
window_max = 80

def has_brain(img_path):
    ###############################################
    # read image
    ############################################### 
    r = pydicom.read_file(img_path.as_posix())
    img = (r.pixel_array * r.RescaleSlope) + r.RescaleIntercept # float64
    img = img.astype(np.int16) # need int16 to store only

    # ###############################################
    # # apply brain window
    # ###############################################
    # bw_img = np.clip(img, window_min, window_max)
    # # img = img * 255 / window_max
    # # img = img.astype(np.uint8)

    ###############################################
    # Get img w/ brain part only
    ###############################################
    brain_img = np.clip(img, 20, 45)
    brain_img[brain_img==45] = 0
    brain_img[brain_img==20] = 0

    # delete small artifacts
    kernel = np.ones((3,3),np.uint8)
    morph = cv2.morphologyEx(brain_img, cv2.MORPH_OPEN, kernel)
    morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
    morph[morph > 0] = 1
    brain_only = brain_img * morph

    has_brain = np.sum(brain_only) > 0
    return has_brain

# Filter brain image
fn = lambda img_path: (img_path.stem, has_brain(img_path))
results = Parallel(n_jobs=8, verbose=1)(delayed(fn)(img_path) for img_path in test_img_paths)

has_brain_df = pd.DataFrame(results, columns=['ImageName', 'BrainPresence'])
has_brain_df.to_csv('data/stage_2_test_hasbrain.csv', index=False)

# merge with test2 metadata
test_metadata = pd.read_csv('data/stage_2_test_metadata.csv')
print('Brain:', has_brain_df[has_brain_df['BrainPresence']].shape[0])
print('No Brain:', has_brain_df[~has_brain_df['BrainPresence']].shape[0])

test_metadata = pd.merge(test_metadata, has_brain_df, left_on='image', right_on='ImageName')
test_metadata.to_csv('data/stage_2_test_metadata.csv', index=False)
