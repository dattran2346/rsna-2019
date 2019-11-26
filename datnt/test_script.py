import os

for f in [3,4]:
    os.system(f'python evaluate.py --modelname seresnext50 --resume version3runs/seresnext50_fold{f}/seresnext50.pth --outputcsv \
            result/datnt_version3_seresnext50_fold{f}.csv --batchsize 32 --datadir ../../rsna/rsna-crop-384-test --trainval test')
