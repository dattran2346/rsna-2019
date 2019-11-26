#!/bin/bash

# mkdir ../data/jpg/stage_2_test_images_prep
# mkdir ../data/jpg/stage_2_test_images_crop
# mkdir ../datnt/stage2result

echo 'MAKE DIRS DONE'

# python 1_convert_dcm_jpg.py

echo 'CONVERT DONE'

# python 2_crop_jpg_384.py

echo 'CROP DONE'

python 3_run_models.py

echo 'INFERENCE DONE'

CUDA_VISIBLE_DEVICES=1 python 4_run_window_cnn.py

echo 'WINDOW DONE'
