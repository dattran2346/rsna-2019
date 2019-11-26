import os
import json
import shutil
import torch
import pickle
import time
import datetime
import numpy as np
import backbone
import regularizer
import argparse
import random
import losses
import pandas as pd

from dataset import RSNADataset
from augmenter import ValAugmenter, HarderAugmenter, FiveCropAugmenter
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from my_scheduler import WarmUpLR
from sklearn.metrics import log_loss
from tqdm import tqdm
from apex import amp

def main(cfg):
    cfg.class_names = cfg.classname.split(",")
    print(cfg)
    
    model = backbone.make_model(cfg.modelname, cfg.class_names)
    old_checkpoint = torch.load(cfg.resume)
    model.load_state_dict(old_checkpoint)

    if cfg.trainval in ['train', 'val']:
        csvfile = os.path.join('csv', f'{cfg.trainval}_fold{cfg.fold}.csv')
    elif cfg.trainval == 'holdout':
        csvfile = os.path.join('csv', 'holdout.csv')
    elif cfg.trainval == 'test':
        csvfile = os.path.join('csv', 'sample_submission.csv')
    elif cfg.trainval in ['train_study', 'val_study']:
        csvfile = os.path.join('csv/nhannt_csv', f'{cfg.trainval[:-6]}_fold{cfg.fold}.csv')
    elif cfg.trainval in ['train_patient', 'val_patient']:
        csvfile = os.path.join('csv/dattq_csv', f'{cfg.trainval[:-8]}_fold{cfg.fold}.csv')
    else:
        raise ValueError('*** trainval ***')
    imagename = pd.read_csv(csvfile)['image'].values
        
    validation_dataset = RSNADataset(dataset_csv_file=csvfile,
                                     class_names=cfg.class_names,
                                     source_image_dir=cfg.datadir,
                                     augmenter=FiveCropAugmenter(cfg.imgsize))
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.batchsize, shuffle=False, num_workers=cfg.workers)

    dtslen = validation_dataset.__len__()
    model = amp.initialize(model, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")
    
    truth, prediction = validate(model, validation_dataloader, dtslen, cfg)
    if cfg.output:
        np.save(cfg.output, prediction)
    if cfg.outputcsv:
        data = [[i,d1,d2,d3,d4,d5,d6] for i,d1,d2,d3,d4,d5,d6 in zip(imagename,prediction[:,0],prediction[:,1],prediction[:,2],prediction[:,3],prediction[:,4],prediction[:,5])]
        sub_df = pd.DataFrame(data=data, columns=['image','any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'])
        sub_df.to_csv(cfg.outputcsv, index=False)
    if cfg.target:
        np.save(cfg.target, truth)

def validate(model, dataloader, dtslen, cfg):
    model.eval()

    y_true_all = np.ndarray(shape=(dtslen, len(cfg.class_names)), dtype=np.float)
    y_pred_all = np.ndarray(shape=(dtslen, len(cfg.class_names)), dtype=np.float)
    idx_runner = 0
    tbar = tqdm(dataloader)
    for _, (x, y_t) in enumerate(tbar):
        with torch.no_grad():
            x = Variable(x).cuda()
            if '_aux' not in cfg.modelname:
                try:
                    y_p = torch.sigmoid(model(x))
                except:
                    bs, ncrops, c, h, w = x.size()
                    y_p = torch.sigmoid(model(x.view(-1, c, h, w))).view(bs, ncrops, -1).mean(1)
            else:
                y_p = torch.sigmoid(model(x)[0]) * 0.8 + torch.sigmoid(model(x)[1]) * 0.2

            y_true = np.vstack(y_t.detach().cpu().numpy())
            y_pred = np.vstack(y_p.detach().cpu().numpy())
            for i in range(len(y_true)):
                y_true_all[idx_runner,:] = y_true[i,:]
                y_pred_all[idx_runner,:] = y_pred[i,:]
                idx_runner += 1

    if cfg.trainval in ['train', 'val', 'train_study', 'val_study', 'train_patient', 'val_patient']:
        ll = []
        for i in range(len(cfg.class_names)):
            yti = y_true_all[:, i]
            ypi = np.clip(y_pred_all[:, i], 1e-15, 1 - 1e-15)
            score = log_loss(yti, ypi)
            if i == 0:
                score *= 2
            ll.append(score)

        print(f'Validation mean LL: {np.mean(ll):.6f}')
        print('Score of each class:', ll)
    return y_true_all, y_pred_all

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--modelname', type=str)
    parser.add_argument('--fold', type=int)
    parser.add_argument('--resume', type=str)
    parser.add_argument('--output', type=str)
    parser.add_argument('--outputcsv', type=str)
    parser.add_argument('--target', type=str)
    parser.add_argument('--datadir', type=str, default='rsna-crop-384')
    parser.add_argument('--classname', type=str, default='any,epidural,intraparenchymal,intraventricular,subarachnoid,subdural')
    parser.add_argument('--batchsize', type=int, default=256)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--imgsize', type=int, default=384)
    parser.add_argument('--trainval', type=str, default='val')
                             
    cfg = parser.parse_args()
                             
    main(cfg)    
