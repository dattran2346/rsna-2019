import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from sklearn.metrics import log_loss
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from rsna_data import RSNADataset, getTransforms
from config import RSNAConfig
from backbone import make_model
from apex import amp
import apex
from shutil import copyfile

torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--net", default='EfficientnetB2', type=str)
parser.add_argument("--folds", nargs="+", type=int)
parser.add_argument("--epochs", nargs="+", type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--quick", default=False, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--fp16", default=True, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--dgx", default=False, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--tta", default=True, type=lambda x: (str(x).lower() == "true"))
args = parser.parse_args()

print('args:',args)

if __name__ == "__main__":
    rsna = RSNAConfig()
    rsna.update(args.net, args.fp16, args.dgx)
    print('cfg:',rsna.conf)

    if not os.path.exists('raw_pred'):
        os.makedirs('raw_pred')

    model = make_model(model_name = rsna.conf.network, num_classes = rsna.conf.num_classes)
    model = model.cuda()

    if args.fp16:
        model = amp.initialize(model, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")

    transforms_train, transforms_test = getTransforms(rsna.conf.size)

    test_df = pd.read_csv(rsna.conf.testset_stage1)
    df = pd.read_csv(rsna.conf.new_trainset)

    if args.quick:
        test_df = test_df.sample(100)

    testset = RSNADataset(df = test_df, root_dir = rsna.conf.stage1_test_dir, transform = transforms_test, datatype='test', tta = args.tta)
    test_loader = DataLoader(testset, batch_size = rsna.conf.batch_size, shuffle = False, num_workers = args.workers)

    for fold in args.folds:
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        valid_df = df.loc[df['fold'] == fold]
        if args.quick:
            valid_df = valid_df.sample(100)
        valid_df = valid_df.reset_index(drop=True)

        valset = RSNADataset(df = valid_df, root_dir = rsna.conf.stage1_train_dir, transform = transforms_test, datatype = 'test', tta = args.tta)
        valid_loader = DataLoader(valset, batch_size = rsna.conf.batch_size, shuffle = False, num_workers = args.workers)

        for epoch in args.epochs:
            print('*'*40 + ' EPOCH {} '.format(epoch) + '*'*40)
            MODEL_CHECKPOINT = 'checkpoints/{}_fold{}_epoch{}.pt'.format(rsna.conf.network, fold, epoch)
            checkpoint = torch.load(MODEL_CHECKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'],strict=True)
            model.eval()

            pred_tmp = []
            if args.tta:
                for inputs1, inputs2 in tqdm(test_loader):
                    inputs1 = inputs1.cuda()
                    inputs2 = inputs2.cuda()
                    with torch.set_grad_enabled(False):
                        outputs1 = torch.sigmoid(model(inputs1))
                        outputs2 = torch.sigmoid(model(inputs2))
                        outputs = torch.add(outputs1, outputs2)
                        outputs = torch.div(outputs,2)
                        if len(outputs.size()) == 1:
                            outputs = torch.unsqueeze(outputs, 0)
                        pred_tmp.append(outputs)
            else:
                for inputs in tqdm(test_loader):
                    inputs = inputs.cuda()
                    with torch.set_grad_enabled(False):
                        outputs = torch.sigmoid(model(inputs))
                        if len(outputs.size()) == 1:
                            outputs = torch.unsqueeze(outputs, 0)
                        pred_tmp.append(outputs)
            pred_tmp = torch.cat(pred_tmp).data.cpu().numpy()
            tmp_df = pd.DataFrame(pred_tmp, columns = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'])
            tmp_df['image'] = test_df.image.values
            tmp_df = tmp_df[['image', 'any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
            tmp_df.to_csv('raw_pred/{}_yp_test_fold{}_epoch{}.csv'.format(args.net, fold, epoch), index=False)

            pred_tmp = []
            del pred_tmp

            pred_tmp = []
            if args.tta:
                for inputs1, inputs2 in tqdm(valid_loader):
                    inputs1 = inputs1.cuda()
                    inputs2 = inputs2.cuda()
                    with torch.set_grad_enabled(False):
                        outputs1 = torch.sigmoid(model(inputs1))
                        outputs2 = torch.sigmoid(model(inputs2))
                        outputs = torch.add(outputs1, outputs2)
                        outputs = torch.div(outputs,2)
                        if len(outputs.size()) == 1:
                            outputs = torch.unsqueeze(outputs, 0)
                        pred_tmp.append(outputs)
            else:
                for inputs in tqdm(valid_loader):
                    inputs = inputs.cuda()
                    with torch.set_grad_enabled(False):
                        outputs = torch.sigmoid(model(inputs))
                        if len(outputs.size()) == 1:
                            outputs = torch.unsqueeze(outputs, 0)
                        pred_tmp.append(outputs)
            pred_tmp = torch.cat(pred_tmp).data.cpu().numpy()
            tmp_df = pd.DataFrame(pred_tmp, columns = ['any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural'])
            tmp_df['image'] = valid_df.image.values
            tmp_df = tmp_df[['image', 'any','epidural','intraparenchymal','intraventricular','subarachnoid','subdural']]
            tmp_df.to_csv('raw_pred/{}_yp_valid_fold{}_epoch{}.csv'.format(args.net, fold, epoch), index=False)

            pred_tmp = []
            del pred_tmp