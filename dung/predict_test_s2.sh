#python3 prepare.py
CUDA_VISIBLE_DEVICES=0 python3 predict_test_stage2.py --net EfficientnetB5 --folds 0 1 2 3 --epochs 11 --workers 8 --fp16 True --dgx True --tta True
CUDA_VISIBLE_DEVICES=0 python3 predict_test_stage2.py --net EfficientnetB5 --folds 4 --epochs 10 --workers 8 --fp16 True --dgx True --tta True
CUDA_VISIBLE_DEVICES=0 python3 predict_test_stage2.py --net EfficientnetB2 --folds 1 2 3 4 --epochs 11 --workers 8 --fp16 True --dgx True --tta True
CUDA_VISIBLE_DEVICES=0 python3 predict_test_stage2.py --net EfficientnetB2 --folds 0 --epochs 5 --workers 8 --fp16 True --dgx True --tta True
CUDA_VISIBLE_DEVICES=0 python3 calibrate_test_s2.py