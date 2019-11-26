import os

for f in [0,1,2,3,4]:
    os.system(f'python evaluate.py --modelname b3 --resume version4runs/b3_fold{f}/b3.pth --outputcsv result/datnt_version4_b3_fold{f}.csv --batchsize 8 \
            --datadir ../../rsna/rsna-crop-384-test --trainval test')
    os.system(f'python evaluate.py --modelname b4 --resume version4runs/b4_fold{f}/b4.pth --outputcsv result/datnt_version4_b4_fold{f}.csv --batchsize 8 \
            --datadir ../../rsna/rsna-crop-384-test --trainval test')
    os.system(f'python evaluate.py --modelname seresnext50 --resume version4runs/seresnext50_fold{f}/seresnext50.pth \
            --outputcsv result/datnt_version3_seresnext50_fold{f}.csv --batchsize 8 \
            --datadir ../../rsna/rsna-crop-384-test --trainval test')
