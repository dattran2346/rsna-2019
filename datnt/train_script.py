import os

# os.system('python main.py --outdir version2runs/inception_fold4 --csvdir csv/nhannt_csv --fold 4 --cutmix True --modelname inceptionv3 --epoch 15')
# for f in [0,1]:
#     os.system(f'python main.py --outdir version2runs/resnet34_fold{f} --csvdir csv/nhannt_csv --fold {f} --cutmix True --modelname resnet34 --epoch 15')
for f in [2]:
    os.system(f'CUDA_VISIBLE_DEVICES=0 python main.py --outdir version4runs/b3_fold{f} --csvdir csv/dattq_csv --fold {f} --cutmix True \
    --modelname b3 --batchsize 28 --lr 1e-4 --epoch 5 --resume version4runs/b3_fold2/b3_e8.pth')
