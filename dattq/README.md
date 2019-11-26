
# How to run

## 1. Setup dataset
- run pre process data for brain, subdural and bony window

```
./process_ct.sh
```

This will create 3 folder under `data/rsna`

```
data/rsna
├── stage_2_test_images_L40_W375
├── stage_2_test_images_L40_W80
└── stage_2_test_images_L75_W215
```

Run `has_brain.py` to get a list of brain image, then merge with `../testset_stage2.csv`

## 2. Copy test csv
cp ../testset_stage2.csv data/rsna/fold

> Note: run nhannt code to generate this file

## 3. Run inference
- Run `se_resnext101_32x4d` on `384x384` image k-fold
- Run `se_resnext50_32x4d` on `256x256` image k-fold

```
./run_inference.sh
```

This will create 10 csv files under `stacking` folder

```
stacking
├── test2_resnext50_32x4d_fold0_x256_tta3.csv
├── test2_resnext50_32x4d_fold1_x256_tta3.csv
├── test2_resnext50_32x4d_fold2_x256_tta3.csv
├── test2_resnext50_32x4d_fold3_x256_tta3.csv
├── test2_resnext50_32x4d_fold4_x256_tta3.csv
├── test2_resnext101_32x4d_fold0_x384_tta3.csv
├── test2_resnext101_32x4d_fold1_x384_tta3.csv
├── test2_resnext101_32x4d_fold2_x384_tta3.csv
├── test2_resnext101_32x4d_fold3_x384_tta3.csv
├── test2_resnext101_32x4d_fold4_x384_tta3.csv
```

## 4. These file will be use by dungnb in stacking stage

