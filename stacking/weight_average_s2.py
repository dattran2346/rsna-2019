import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import torch
import torch.nn as nn
from utils import *
from torch.utils.data import DataLoader, Dataset

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--folds", nargs="+", type=int)
args = parser.parse_args()

test_dict = {
      0: {'../dung/calib_pred/EfficientnetB5_yp_test_s2_fold0_epoch11.csv': 0.2,
          '../output_stage2/datnt_version3_seresnext50_fold0.csv': 0.2,
          '../output_stage2/datnt_version4_b4_fold0.csv': 0.1,
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold0.csv': 0.25,
          '../dattq/stacking/test2_resnext101_32x4d_fold0_x384_tta3.csv': 0.25},

      1: {'../dung/calib_pred/EfficientnetB5_yp_test_s2_fold1_epoch11.csv': 0.2,
          '../output_stage2/datnt_version3_seresnext50_fold1.csv': 0.2,
          '../output_stage2/datnt_version4_b4_fold1.csv': 0.1,
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold1.csv': 0.25,
          '../dattq/stacking/test2_resnext101_32x4d_fold1_x384_tta3.csv': 0.25},

      2: {'../dung/calib_pred/EfficientnetB5_yp_test_s2_fold2_epoch11.csv': 0.2,
          '../output_stage2/datnt_version3_seresnext50_fold2.csv': 0.2,
          '../output_stage2/datnt_version4_b4_fold2.csv': 0.1,
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold2.csv': 0.25,
          '../dattq/stacking/test2_resnext101_32x4d_fold2_x384_tta3.csv': 0.25},

      3: {'../dung/calib_pred/EfficientnetB5_yp_test_s2_fold3_epoch11.csv': 0.2,
          '../output_stage2/datnt_version3_seresnext50_fold3.csv': 0.2,
          '../output_stage2/datnt_version4_b4_fold3.csv': 0.1,
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold3.csv': 0.25,
          '../dattq/stacking/test2_resnext101_32x4d_fold3_x384_tta3.csv': 0.25},

      4: {'../dung/calib_pred/EfficientnetB5_yp_test_s2_fold4_epoch10.csv': 0.2,
          '../output_stage2/datnt_version3_seresnext50_fold4.csv': 0.2,
          '../output_stage2/datnt_version4_b4_fold4.csv': 0.1,
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold4.csv': 0.25,
          '../dattq/stacking/test2_resnext101_32x4d_fold4_x384_tta3.csv': 0.25},
}

print('args:',args)
if __name__ == "__main__":
    test_df = pd.read_csv('../dung/dataset/testset_stage2.csv')
    test_df = test_df.sort_values(by='image', ascending=False).reset_index(drop=True)
    
    ytest = np.zeros((len(test_df), 6), dtype=np.float64)
    for fold in args.folds:
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        for raw_file, weight in test_dict[fold].items():
            tmp_df = pd.read_csv(raw_file)
            tmp_df = tmp_df.sort_values(by='image', ascending=False).reset_index(drop=True)
            pred = tmp_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']].values
            ytest += weight*pred
    ytest /= len(args.folds)

    IDs = []
    Labels = []
    tmp_df = pd.DataFrame(ytest, columns = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'])
    tmp_df['image'] = test_df.image.values
    for _, row in tmp_df.iterrows():
        for diagnostic in ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']:
            IDs.append(row['image'] + '_' + diagnostic)
            Labels.append(row[diagnostic])
    submission_df = pd.DataFrame()
    submission_df['ID'] = np.array(IDs)
    submission_df['Label'] = np.array(Labels)
    submission_df.to_csv('../weight_average_submission.csv', index=False)
    print(submission_df.head(10))