import os

for f in [0,1,2,3,4]:
    os.system(f'CUDA_VISIBLE_DEVICES=1 python evaluate.py --modelname b3 --resume version4runs/b3_fold{f}/b3.pth --outputcsv stage2result/datnt_version4_b3_fold{f}.csv --batchsize 64 \
            --datadir ../data/jpg/stage_2_test_images_crop --trainval test --workers 32')
    os.system(f'CUDA_VISIBLE_DEVICES=1 python evaluate.py --modelname b4 --resume version4runs/b4_fold{f}/b4.pth --outputcsv stage2result/datnt_version4_b4_fold{f}.csv --batchsize 64 \
            --datadir ../data/jpg/stage_2_test_images_crop --trainval test --workers 32')
    os.system(f'CUDA_VISIBLE_DEVICES=1 python evaluate.py --modelname seresnext50 --resume version3runs/seresnext50_fold{f}/seresnext50.pth \
            --outputcsv stage2result/datnt_version3_seresnext50_fold{f}.csv --batchsize 64 \
            --datadir ../data/jpg/stage_2_test_images_crop --trainval test --workers 32')
