# brain
# python3 process_ct.py --window-level 40 --window-width 80 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 40 --window-width 80 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'

# subdural
# python3 process_ct.py --window-level 100 --window-width 300 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 100 --window-width 300 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'
# subdural window, from md.ai
# python3 process_ct.py --window-level 75 --window-width 215 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 75 --window-width 215 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'

# bone
# python3 process_ct.py --window-level 600 --window-width 2800 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 600 --window-width 2800 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'

# soft tissue
# python3 process_ct.py --window-level 60 --window-width 400 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 60 --window-width 400 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'
# neck soft tissue, from md.ai
# python3 process_ct.py --window-level 40 --window-width 375 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 40 --window-width 375 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'

# stroke window 1
# python3 process_ct.py --window-level 40 --window-width 40 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 40 --window-width 40 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'

# stroke window 2
# python3 process_ct.py --window-level 32 --window-width 8 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_test_images'
# python3 process_ct.py --window-level 32 --window-width 8 --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin/stage_1_train_images'


## Preprocess data stage2
# brain
python3 process_ct.py --window-level 40 --window-width 80 --data-dir '../data/dicom/stage_2_test_images' --output-dir 'data/rsna'
# subdural
python3 process_ct.py --window-level 75 --window-width 215 --data-dir '../data/dicom/stage_2_test_images' --output-dir 'data/rsna'
# bony
python3 process_ct.py --window-level 600 --window-width 2800 --data-dir '../data/dicom/stage_2_test_images' --output-dir 'data/rsna'

