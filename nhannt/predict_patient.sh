python src/main_rsna_3d.py -et  TTA True MODEL_NAME "'$1'" SESS_NAME "'$1_bigru_patient_fold$2'" \
RECUR_TYPE "'bigru'" FOLD $2 RESUME "'rsna_weights/best_$1_bigru_patient_fold$2.pth'" \
DATA_DIR "'$3'" IMG_SIZE $4 SPLIT "'patient'"