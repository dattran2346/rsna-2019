### Instruction to run inference

**Step 1. Create preprocessed data from DICOM**

1a. Use notebooks *___ Data Preprocessing ___* to generate JPG files.

1b. Use notebook *___ Crop IMG to 384 ___* to crop JPG image to sz 384.

1c. Get modified sample submission CSV to load image names. I get my csv file from
teamate DungNB. (refer line 15 file dungnb/config.py self.conf.testset_stage2 = 'dataset/testset_stage2.csv')
 
**Step 2. Run inference models**

Use script *validation_script.py* to run inference. This script to call to file *evaluation.py*. To run inference, use flag *--modelname* to get name of model architecture (for example: seresnext50, b3, b4), use flag *--resume* to specify where to load checkpoint (for example: *version3runs/seresnext50_fold0/seresnext50.pth*), use flag *--outputcsv* to specify where to output csv file of inference result, set flag *--trainval* to *test*.

**Step 3. Run window_cnn models to improve 2D models**

Use notebook *___ Window ___* to run window_cnn to improve 2D models output. Set variable *fold* to specify fold of architecture load checkpoint and window CNN checkpoint (will be from *0* to *4*), set *valtest* variable to *test* to run inference, set *studypatient* variable to *study* for models in *version3runs* folder or *patient* for models in *version4runs* and set *csv_name* to output csv file from inference models.

To run window cnn, a **metadata CSV** is needed. I got my metadata CSV file from my teammate NguyenThanhNhan.

**Step 4. Stack models**

I give my k-fold validation CSVs and inference CSVs file to my teammate DungNB to build stacking metamodels.