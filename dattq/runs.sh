# Train k-fold se_resnext_32x4d
# ./train_remotely.sh medical 0 --backbone 'resnext50_32x4d' --att 'se' --fold 1 --epochs 40 --warmup 4 --batch-size 8 --gds 8 --image-size 256 --checkname resnext50_32x4d_fold1_x256_bilstm --resume runs/resnext50_32x4d_fold1_x256_bilstm/checkpoint.pth
# ./train_remotely.sh medical 2 --backbone 'resnext50_32x4d' --att 'se' --fold 2 --epochs 40 --warmup 4 --batch-size 8 --gds 8 --image-size 256 --checkname resnext50_32x4d_fold2_x256_bilstm --resume runs/resnext50_32x4d_fold2_x256_bilstm/checkpoint.pth

# this is good, but
# ./train_remotely.sh gcp 0 --backbone 'resnext50_32x4d' --att 'se' --fold 3 --epochs 40 --warmup 4 --batch-size 16 --gds 4 --image-size 256 --checkname resnext50_32x4d_fold3_x256_bilstm --resume runs/resnext50_32x4d_fold3_x256_bilstm/checkpoint.pth

# ./train_remotely.sh medical 3 --backbone 'resnext50_32x4d' --att 'se' --fold 4 --batch-size 8 --gds 8 --image-size 256 --checkname resnext50_32x4d_fold4_x256_bilstm --resume runs/resnext50_32x4d_fold4_x256_bilstm/checkpoint.pth

# Train for 30 epoch only
# ./train_remotely.sh medical 0 --backbone 'resnext50_32x4d' --att 'se' --fold 1 --epochs 30 --warmup 3 --batch-size 8 --gds 8 --image-size 256 --checkname resnext50_32x4d_fold1_x256_bilstm --resume runs/resnext50_32x4d_fold1_x256_bilstm/checkpoint.pth
# ./train_remotely.sh medical 2 --backbone 'resnext50_32x4d' --att 'se' --fold 2 --epochs 30 --warmup 3 --batch-size 8 --gds 8 --image-size 256 --checkname resnext50_32x4d_fold2_x256_bilstm --resume runs/resnext50_32x4d_fold2_x256_bilstm/checkpoint.pth
# ./train_remotely.sh medical 2 --backbone 'resnext50_32x4d' --att 'se' --fold 3 --epochs 30 --warmup 3 --batch-size 4 --gds 16 --image-size 256 --checkname resnext50_32x4d_fold3_x256_bilstm --resume runs/resnext50_32x4d_fold3_x256_bilstm/checkpoint.pth


# train again fold 3 with temporal flip
# ./train_remotely.sh gcp 0 --backbone 'resnext50_32x4d' --att 'se' --fold 3 --epochs 30 --warmup 3 --batch-size 16 --gds 4 --image-size 256 --checkname resnext50_32x4d_fold3_x256_bilstm_tflip_30e --resume runs/resnext50_32x4d_fold3_x256_bilstm_tflip_30e/checkpoint.pth
# fold 4 currently perform too bad 0.87, try to retrain w temporal flip
# ./train_remotely.sh gcp 0 --backbone 'resnext50_32x4d' --att 'se' --fold 4 --epochs 30 --warmup 3 --batch-size 16 --gds 4 --image-size 256 --checkname resnext50_32x4d_fold4_x256_bilstm.gcp --resume runs/resnext50_32x4d_fold4_x256_bilstm.gcp/checkpoint.pth
# ./train_remotely.sh gcp 0 --backbone 'resnext50_32x4d' --att 'se' --fold 1 --epochs 30 --warmup 3 --batch-size 16 --gds 4 --image-size 256 --checkname resnext50_32x4d_fold1_x256_bilstm.gcp --resume runs/resnext50_32x4d_fold1_x256_bilstm.gcp/checkpoint.pth
# ./train_remotely.sh gcp 0 --backbone 'resnext50_32x4d' --att 'se' --fold 2 --epochs 30 --warmup 3 --batch-size 16 --gds 4 --image-size 256 --checkname resnext50_32x4d_fold2_x256_bilstm.gcp --resume runs/resnext50_32x4d_fold2_x256_bilstm.gcp/checkpoint.pth


## train seresnext101 x384
./train_remotely.sh gcp2 0 --backbone 'resnext101_32x4d' --att 'se' --fold 0 --epochs 30 --warmup 3 --batch-size 4 --gds 16 --image-size 384 --checkname resnext101_32x4d_fold0_x384_bilstm --resume runs/resnext101_32x4d_fold0_x384_bilstm/checkpoint.pth
# ./train_remotely.sh dgx1 2 --backbone 'resnext101_32x4d' --att 'se' --fold 1 --epochs 30 --warmup 3 --batch-size 8 --gds 8 --image-size 384 --checkname resnext101_32x4d_fold1_x384_bilstm --resume runs/resnext101_32x4d_fold1_x384_bilstm/checkpoint.pth
## ./train_remotely.sh dgx1 3 --backbone 'resnext101_32x4d' --att 'se' --fold 2 --epochs 30 --warmup 3 --batch-size 8 --gds 8 --image-size 384 --checkname resnext101_32x4d_fold2_x384_bilstm --resume runs/resnext101_32x4d_fold2_x384_bilstm/checkpoint.pth
## ./train_remotely.sh dgx1 1 --backbone 'resnext101_32x4d' --att 'se' --fold 3 --epochs 30 --warmup 3 --batch-size 8 --gds 8 --image-size 384 --checkname resnext101_32x4d_fold3_x384_bilstm --resume runs/resnext101_32x4d_fold3_x384_bilstm/checkpoint.pth
## ./train_remotely.sh dgx1 0 --backbone 'resnext101_32x4d' --att 'se' --fold 4 --epochs 30 --warmup 3 --batch-size 8 --gds 8 --image-size 384 --checkname resnext101_32x4d_fold4_x384_bilstm --resume runs/resnext101_32x4d_fold4_x384_bilstm/checkpoint.pth
