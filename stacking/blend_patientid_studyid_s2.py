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

studyid_test_dict = {
      0: ['../dung/calib_pred/EfficientnetB2_yp_test_s2_fold0_epoch5.csv',
          '../dung/calib_pred/EfficientnetB5_yp_test_s2_fold0_epoch11.csv',
          '../output_stage2/datnt_version3_seresnext50_fold0.csv',
          '../nghia/outputs/test_exp25_fold0_lstm.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext50_32x4d_a2_b4_bilstm_fold0.csv',
          '../nhannt/rsna_outputs_stage2/test_se_resnext101_32x4d_bilstm_fold0.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold0.csv'],

      1: ['../dung/calib_pred/EfficientnetB2_yp_test_s2_fold1_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_test_s2_fold1_epoch11.csv',
          '../output_stage2/datnt_version3_seresnext50_fold1.csv',
          '../nghia/outputs/test_exp25_fold1_lstm.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext50_32x4d_a2_b4_bilstm_fold1.csv',
          '../nhannt/rsna_outputs_stage2/test_se_resnext101_32x4d_bilstm_fold1.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold1.csv'],

      2: ['../dung/calib_pred/EfficientnetB2_yp_test_s2_fold2_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_test_s2_fold2_epoch11.csv',
          '../output_stage2/datnt_version3_seresnext50_fold2.csv',
          '../nghia/outputs/test_exp25_fold2_lstm.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext50_32x4d_a2_b4_bilstm_fold2.csv',
          '../nhannt/rsna_outputs_stage2/test_se_resnext101_32x4d_bilstm_fold2.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold2.csv'],

      3: ['../dung/calib_pred/EfficientnetB2_yp_test_s2_fold3_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_test_s2_fold3_epoch11.csv',
          '../output_stage2/datnt_version3_seresnext50_fold3.csv',
          '../nghia/outputs/test_exp25_fold3_lstm.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext50_32x4d_a2_b4_bilstm_fold3.csv',
          '../nhannt/rsna_outputs_stage2/test_se_resnext101_32x4d_bilstm_fold3.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold3.csv'],

      4: ['../dung/calib_pred/EfficientnetB2_yp_test_s2_fold4_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_test_s2_fold4_epoch10.csv',
          '../output_stage2/datnt_version3_seresnext50_fold4.csv',
          '../nghia/outputs/test_exp25_fold4_lstm.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext50_32x4d_a2_b4_bilstm_fold4.csv',
          '../nhannt/rsna_outputs_stage2/test_se_resnext101_32x4d_bilstm_fold4.csv',
          '../nhannt/rsna_outputs_stage2/test_blseresnext101_32x4d_a2_b4_bilstm_fold4.csv'],
}

patientid_train_dict = {
    0: ['../output_stage2/datnt_version4_b3_fold0.csv',
        '../output_stage2/datnt_version4_b4_fold0.csv',
        '../nhannt/rsna_outputs_stage2/test_tf_efficientnet_b3_bigru_patient_fold0.csv',
        '../dattq/stacking/test2_resnext50_32x4d_fold0_x256_tta3.csv',
        '../dattq/stacking/test2_resnext101_32x4d_fold0_x384_tta3.csv'],

    1: ['../output_stage2/datnt_version4_b3_fold1.csv',
        '../output_stage2/datnt_version4_b4_fold1.csv',
        '../nhannt/rsna_outputs_stage2/test_tf_efficientnet_b3_bigru_patient_fold1.csv',
        '../dattq/stacking/test2_resnext50_32x4d_fold1_x256_tta3.csv',
        '../dattq/stacking/test2_resnext101_32x4d_fold1_x384_tta3.csv'],

    2: ['../output_stage2/datnt_version4_b3_fold2.csv',
        '../output_stage2/datnt_version4_b4_fold2.csv',
        '../nhannt/rsna_outputs_stage2/test_tf_efficientnet_b3_bigru_patient_fold2.csv',
        '../dattq/stacking/test2_resnext50_32x4d_fold2_x256_tta3.csv',
        '../dattq/stacking/test2_resnext101_32x4d_fold2_x384_tta3.csv'],

    3: ['../output_stage2/datnt_version4_b3_fold3.csv',
        '../output_stage2/datnt_version4_b4_fold3.csv',
        '../nhannt/rsna_outputs_stage2/test_tf_efficientnet_b3_bigru_patient_fold3.csv',
        '../dattq/stacking/test2_resnext50_32x4d_fold3_x256_tta3.csv',
        '../dattq/stacking/test2_resnext101_32x4d_fold3_x384_tta3.csv'],

    4: ['../output_stage2/datnt_version4_b3_fold4.csv',
        '../output_stage2/datnt_version4_b4_fold4.csv',
        '../nhannt/rsna_outputs_stage2/test_tf_efficientnet_b3_bigru_patient_fold4.csv',
        '../dattq/stacking/test2_resnext50_32x4d_fold4_x256_tta3.csv',
        '../dattq/stacking/test2_resnext101_32x4d_fold4_x384_tta3.csv'],
}

print('args:',args)
if __name__ == "__main__":
    test_df = pd.read_csv('../dung/dataset/testset_stage2.csv')
    test_df = test_df.sort_values(by='image', ascending=False).reset_index(drop=True)
    
    ytest_studtyid = np.zeros((len(test_df), 6), dtype=np.float64)
    for fold in args.folds:
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        xtest = np.array([], dtype=np.float64).reshape(len(test_df),0)
        for raw_file in studyid_test_dict[fold]:
            tmp_df = pd.read_csv(raw_file)
            tmp_df = tmp_df.sort_values(by='image', ascending=False).reset_index(drop=True)
            pred = tmp_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']].values
            xtest = np.hstack((xtest, pred))
        
        model = StackingModel1(xtest.shape[1])
        model = model.cuda()

        testset = StackingDataset(x = xtest, y = None, datatype='test')
        test_loader = DataLoader(testset, batch_size = 128, shuffle = False, num_workers = 0)

        for sub_fold in range(10):
            MODEL_CHECKPOINT = 'checkpoints_study_id/fold{}_sub{}.pt'.format(fold, sub_fold)
            checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cuda:0')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            pred_tmp = []
            for inputs in test_loader:
                inputs = inputs.cuda()
                with torch.set_grad_enabled(False):
                    outputs = torch.sigmoid(model(inputs))
                    if len(outputs.size()) == 1:
                        outputs = torch.unsqueeze(outputs, 0)
                    pred_tmp.append(outputs)
            pred_tmp = torch.cat(pred_tmp).data.cpu().numpy()
            ytest_studtyid += pred_tmp
    ytest_studtyid /= float(len(args.folds)*10)


    ytest_patientid = np.zeros((len(test_df), 6), dtype=np.float64)
    for fold in args.folds:
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        xtest = np.array([], dtype=np.float64).reshape(len(test_df),0)
        for raw_file in patientid_train_dict[fold]:
            tmp_df = pd.read_csv(raw_file)
            tmp_df = tmp_df.sort_values(by='image', ascending=False).reset_index(drop=True)
            pred = tmp_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']].values
            xtest = np.hstack((xtest, pred))
        
        model = StackingModel1(xtest.shape[1])
        model = model.cuda()

        testset = StackingDataset(x = xtest, y = None, datatype='test')
        test_loader = DataLoader(testset, batch_size = 64, shuffle = False, num_workers = 0)

        for sub_fold in range(10):
            MODEL_CHECKPOINT = 'checkpoints_patients_id/fold{}_sub{}.pt'.format(fold, sub_fold)
            checkpoint = torch.load(MODEL_CHECKPOINT, map_location='cuda:0')
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()

            pred_tmp = []
            for inputs in test_loader:
                inputs = inputs.cuda()
                with torch.set_grad_enabled(False):
                    outputs = torch.sigmoid(model(inputs))
                    if len(outputs.size()) == 1:
                        outputs = torch.unsqueeze(outputs, 0)
                    pred_tmp.append(outputs)
            pred_tmp = torch.cat(pred_tmp).data.cpu().numpy()
            ytest_patientid += pred_tmp
    ytest_patientid /= float(len(args.folds)*10)

    ytest = 0.5*ytest_studtyid + 0.5*ytest_patientid

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
    submission_df.to_csv('../stacking_submission.csv', index=False)
    print(submission_df.head(10))