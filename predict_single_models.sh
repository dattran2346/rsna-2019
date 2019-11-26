cd nhannt
./predict.sh
cd ..

cd nghia
# Octave Resnet50
./infer.sh 0 --test
./infer.sh 1 --test
./infer.sh 2 --test
./infer.sh 3 --test
./infer.sh 4 --test
cd ..

cd dattq
# run on 0th gpu
./run_inference.sh 0
cd ..

cd dung
bash predict_test_s2.sh
cd ..

cd datnt
python 3_run_models.py
python 4_run_windows_cnn.py
cd ..
