## Make dirs
cd data
mkdir npy
mkdir npy/stage_1_train_images
mkdir npy/stage_2_test_images
mkdir jpg
mkdir jpg/stage_2_test_images_prep
mkdir jpg/stage_2_test_images_crop
cd ../dattq
mkdir -p data/rsna/fold
cd ../datnt
mkdir stage2result
cd ..

## Convert DICOM to npy and extract metadata (NhanNT, NghiaNT)
python convert_dicom_to_npy.py --data-dir ./data/dicom/stage_2_test_images --output-dir ./data/npy/stage_2_test_images
python extract_metadata_from_dicom.py

## dattq
## Create separate windows from DICOM and save as JPG (DatTQ)
# brain
python process_ct.py --window-level 40 --window-width 80 --data-dir 'data/dicom/stage_2_test_images' --output-dir 'data/jpg'
# subdural
python process_ct.py --window-level 75 --window-width 215 --data-dir 'data/dicom/stage_2_test_images' --output-dir 'data/jpg'
# bony
python process_ct.py --window-level 600 --window-width 2800 --data-dir 'data/dicom/stage_2_test_images' --output-dir 'data/jpg'

# Symbolic link
ln -sf $(pwd)/data/jpg/stage_2_test_images_L40_W80 dattq/data/rsna/
ln -sf $(pwd)/data/jpg/stage_2_test_images_L75_W215 dattq/data/rsna/
ln -sf $(pwd)/data/jpg/stage_2_test_images_L600_W2800 dattq/data/rsna/

# Has brain filter
python has_brain.py 
ln -sf $(pwd)/data/stage_2_test_metadata.csv dattq/data/rsna/fold/testset_stage2.csv

## Convert DICOM to JPG and crop (DatNT)
python convert_dicom_to_jpg.py
python crop_jpg_384.py
