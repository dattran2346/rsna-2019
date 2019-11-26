python src/main_rsna_3d.py -et  TTA True MODEL_NAME "'$1'" SESS_NAME "'$1_bilstm_fold$2'" \
FOLD $2 RESUME "'rsna_weights/best_$1_bilstm_fold$2.pth'" DATA_DIR "'$3'" IMG_SIZE $4 