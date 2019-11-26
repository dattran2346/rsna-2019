# RSNA 2019 13th place solution

Credit: [@andy2709](https://github.com/nhannguyen2709), [lego1st](https://github.com/Lego1st), [DungNB](https://github.com/dungnb1333), [DatNT](https://github.com/moewiee) 

## Instruction

- download stage 2 test data and unzip to data/dicom/stage_2_test_images/ 
- run pre-processing script

```
$ chmod +x preprocess.sh
$ ./preprocess.sh
```

- Predict all single models

```
$ chmod +x predict_single_models.sh
$ ./predict_single_models.sh
```

- Run stacking
```
$ bash stacking_predict.sh
```

### final submission is 2 files stacking_submission.csv and weight_average_submission.csv in root
