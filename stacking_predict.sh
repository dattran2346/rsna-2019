cd stacking
python3 blend_patientid_studyid_s2.py --folds 0 1 2 3 4
python3 weight_average_s2.py --folds 0 1 2 3 4
cd ..