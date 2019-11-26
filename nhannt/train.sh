CUDA_VISIBLE_DEVICES=$8 python src/main_rsna_3d.py MODEL_NAME "'$1'" SESS_NAME "'$1_bilstm_fold$2'" FOLD $2 DATA_DIR "'$3'" \
BATCH_SIZE $4 GD_STEPS $5 BASE_LR $6 TTA False FILTER_NO_BRAIN True \
IMG_SIZE $7 SPLIT "'study'"