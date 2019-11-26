import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
import pickle
import random

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable

from cyclic_scheduler import CyclicLRWithRestarts
from utils import *
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold, MultilabelStratifiedShuffleSplit,RepeatedMultilabelStratifiedKFold

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--folds", nargs="+", type=int)
parser.add_argument("--epochs", default=55, type=int)
parser.add_argument("--patience", default=30, type=int)
parser.add_argument("--nfolds", default=5, type=int)
parser.add_argument("--nrepeats", default=2, type=int)
args = parser.parse_args()

model_train_dict = {
      0: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold0_epoch5.csv',
          '../dung/calib_pred/EfficientnetB5_yp_valid_fold0_epoch11.csv',
          '../datnt/windowed_csv/val/datnt_version3_seresnext50_fold0.csv',
          '../nghia/raw_pred_s1/valid_exp25_fold0_lstm.csv',
          '../nhannt/raw_pred/val_blseresnext50_32x4d_a2_b4_bilstm_fold0.csv',
          '../nhannt/raw_pred/val_se_resnext101_32x4d_bilstm_fold0.csv',
          '../nhannt/raw_pred/val_blseresnext101_32x4d_a2_b4_bilstm_fold0.csv'],

      1: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold1_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_valid_fold1_epoch11.csv',
          '../datnt/windowed_csv/val/datnt_version3_seresnext50_fold1.csv',
          '../nghia/raw_pred_s1/valid_exp25_fold1_lstm.csv',
          '../nhannt/raw_pred/val_blseresnext50_32x4d_a2_b4_bilstm_fold1.csv',
          '../nhannt/raw_pred/val_se_resnext101_32x4d_bilstm_fold1.csv',
          '../nhannt/raw_pred/val_blseresnext101_32x4d_a2_b4_bilstm_fold1.csv'],

      2: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold2_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_valid_fold2_epoch11.csv',
          '../datnt/windowed_csv/val/datnt_version3_seresnext50_fold2.csv',
          '../nghia/raw_pred_s1/valid_exp25_fold2_lstm.csv',
          '../nhannt/raw_pred/val_blseresnext50_32x4d_a2_b4_bilstm_fold2.csv',
          '../nhannt/raw_pred/val_se_resnext101_32x4d_bilstm_fold2.csv',
          '../nhannt/raw_pred/val_blseresnext101_32x4d_a2_b4_bilstm_fold2.csv'],

      3: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold3_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_valid_fold3_epoch11.csv',
          '../datnt/windowed_csv/val/datnt_version3_seresnext50_fold3.csv',
          '../nghia/raw_pred_s1/valid_exp25_fold3_lstm.csv',
          '../nhannt/raw_pred/val_blseresnext50_32x4d_a2_b4_bilstm_fold3.csv',
          '../nhannt/raw_pred/val_se_resnext101_32x4d_bilstm_fold3.csv',
          '../nhannt/raw_pred/val_blseresnext101_32x4d_a2_b4_bilstm_fold3.csv'],

      4: ['../dung/calib_pred/EfficientnetB2_yp_valid_fold4_epoch11.csv',
          '../dung/calib_pred/EfficientnetB5_yp_valid_fold4_epoch10.csv',
          '../datnt/windowed_csv/val/datnt_version3_seresnext50_fold4.csv',
          '../nghia/raw_pred_s1/valid_exp25_fold4_lstm.csv',
          '../nhannt/raw_pred/val_blseresnext50_32x4d_a2_b4_bilstm_fold4.csv',
          '../nhannt/raw_pred/val_se_resnext101_32x4d_bilstm_fold4.csv',
          '../nhannt/raw_pred/val_blseresnext101_32x4d_a2_b4_bilstm_fold4.csv'],
}

print('args:',args)
if __name__ == "__main__":
    if not os.path.exists('checkpoints_study_id'):
        os.makedirs('checkpoints_study_id')

    train_criterion = nn.BCEWithLogitsLoss()
    weight_tensor = torch.FloatTensor([2., 1., 1., 1., 1., 1.]).cuda()
    val_criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)
    
    df = pd.read_csv('../dung/dataset/trainset_nhannt.csv')
    LOGS = 'checkpoints_study_id/training_log.csv'
    for fold in args.folds:
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        str_tmp = '*'*40 + ' FOLD {} '.format(fold) + '*'*40 + '\n'
        valid_df = df.loc[df['fold'] == fold].sort_values(by='image', ascending=False).reset_index(drop=True)
        yt = valid_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']].values
        avg_pred = np.zeros_like(yt).astype(np.float64)
        xt = np.array([], dtype=np.float64).reshape(len(valid_df),0)
        for raw_file in model_train_dict[fold]:
            tmp_df = pd.read_csv(raw_file)
            tmp_df = tmp_df.sort_values(by='image', ascending=False).reset_index(drop=True)
            pred = tmp_df[['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']].values
            xt = np.hstack((xt, pred))
            print(weightLogLoss(yt, pred),raw_file)
            str_tmp += str(weightLogLoss(yt, pred)) + '\t' + raw_file + '\n'
            avg_pred += pred
        avg_pred /= float(len(model_train_dict[fold]))
        average_loss = weightLogLoss(yt, avg_pred)
        print('AVERAGE LOSS: {}'.format(average_loss))
        str_tmp += 'AVERAGE LOSS: {}\n'.format(average_loss)

        stacking_loss = []
        rmskf = RepeatedMultilabelStratifiedKFold(n_splits=args.nfolds, n_repeats=args.nrepeats, random_state=8)
        for sub_fold, (train_index, test_index) in enumerate(rmskf.split(xt, yt)):
            print("TRAIN SIZE: {} TEST SIZE: {}".format(len(train_index), len(test_index)))
            x_train, x_val = xt[train_index], xt[test_index]
            y_train, y_val = yt[train_index], yt[test_index]

            trainset = StackingDataset(x = x_train, y = y_train, datatype='train')
            train_loader = DataLoader(trainset, batch_size=128, shuffle = True, num_workers = 8)

            valset = StackingDataset(x = x_val, y = y_val, datatype='train')
            valid_loader = DataLoader(valset, batch_size = 128, shuffle = False, num_workers = 8)

            model = StackingModel1(x_train.shape[1])
            model = model.cuda()

            optimizer = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-4)
            scheduler = CyclicLRWithRestarts(
                optimizer = optimizer,
                batch_size = 128,
                epoch_size = len(train_loader.dataset),
                restart_period=5,
                t_mult=1.2,
                policy="cosine")

            MODEL_CHECKPOINT = 'checkpoints_study_id/fold{}_sub{}.pt'.format(fold, sub_fold)
            TRAINING_LOG = 'checkpoints_study_id/training_log_fold{}_sub{}.csv'.format(fold, sub_fold)

            if os.path.isfile(TRAINING_LOG):
                os.remove(TRAINING_LOG)

            valid_loss_min = np.Inf

            pat = 0
            for epoch in range(args.epochs):
                train_loss = 0.0
                model.train()

                for inputs, labels in train_loader:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.5, True)
                    inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        if len(outputs.size()) == 1:
                                outputs = torch.unsqueeze(outputs, 0)
                        loss = mixup_criterion(train_criterion, outputs, labels_a, labels_b, lam)

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        scheduler.batch_step()

                    train_loss += loss.item() * inputs.size(0)
                train_loss = train_loss / len(train_loader.dataset)

                scheduler.step()

                model.eval()
                valid_loss = 0.0

                for inputs, labels in valid_loader:
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        if len(outputs.size()) == 1:
                                outputs = torch.unsqueeze(outputs, 0)
                        loss = val_criterion(outputs, labels)
                    valid_loss += loss.item() * inputs.size(0)
                valid_loss = valid_loss / len(valid_loader.dataset)

                print('Epoch: {:2d}/{:2d} | Train loss: {:.5f} | Val loss: {}'.format(epoch, args.epochs-1, train_loss, valid_loss))
                log_file = open(TRAINING_LOG, 'a')
                log_file.write('Epoch: {:2d}/{:2d} | Train loss: {:.5f} | Val loss: {:.5f}\n'.format(epoch, args.epochs-1, train_loss, valid_loss))
                log_file.close()

                if valid_loss <= valid_loss_min:
                    print('Valid log loss improved from {:.5f} to {:.5f} saving model to {}'.format(valid_loss_min, valid_loss, MODEL_CHECKPOINT))
                    valid_loss_min = valid_loss
                    pat = 0
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'valid_loss_min': valid_loss_min,
                    }, MODEL_CHECKPOINT)
                else:
                    pat += 1

                if pat == args.patience or epoch == args.epochs-1:
                    stacking_loss.append(valid_loss_min)
                    print('SUB FOLD: {} AVERAGE LOSS: {} STACKING LOSS: {}'.format(sub_fold, average_loss, np.mean(stacking_loss)))
                    break
        print('FOLD: {} AVERAGE LOSS: {} STACKING LOSS: {}'.format(fold, average_loss, np.mean(stacking_loss)))
        str_tmp += 'STACKING LOSS: {}\n'.format(np.mean(stacking_loss))
        log_file = open(LOGS, 'a')
        log_file.write(str_tmp)
        log_file.close()