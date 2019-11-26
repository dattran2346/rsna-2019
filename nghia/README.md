# RSNA Intracranial Hemorrhage Detection

## Config file

**Main config:** 

./config.py

**Custom config for experiment:** 

./expconfigs/exp{N_EXP}_fold{N_FOLD}.yaml *(Example: 
exp25_fold0_lstm.yaml)*

## Run test inference script

```
# Run test for fold 0, 1, 2, 3, 4
./infer.sh 0 --test
./infer.sh 1 --test
./infer.sh 2 --test
./infer.sh 3 --test
./infer.sh 4 --test

```