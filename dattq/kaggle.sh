

# #############################################
# # FIRST RUN PREPROCESS HU VALUE FOR NGHIA AND NHANNT
# #############################################
# python3 process_hu_ct.py --data-dir '/mnt/DATA/kaggle/rsna-intracranial-hemorrhage-detection/origin' --output-dir $RSNA_DIR

#############################################
# INFERENCE
#############################################
python3 inference.py --data-dir $RSNA_DIR --mix-window 3 --backbone 'resnext101_32x4d' --att 'se' --image-size 384 --batch-size 128 --input-level "per-study" --infer-sigmoid --fold 0 --resume models_collections/resnext101_32x4d_fold0_x384_bilstm/best_model.pth --checkname test_resnext101_32x4d_fold0_x384_tta3 --tta 3

############################################
# The output of this script will be go to dungnb stacking
############################################