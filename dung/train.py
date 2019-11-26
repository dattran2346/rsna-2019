import argparse
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import time
from apex import amp
import apex
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.autograd import Variable

from utils import mixup_data, mixup_criterion
from rsna_data import RSNADataset, getTransforms
from config import RSNAConfig
from backbone import make_model
from cyclic_scheduler import CyclicLRWithRestarts
from utils import seed_everything
# from apex.optimizers import FusedAdam

seed_everything(seed=8)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--net", default='EfficientnetB4', type=str)
parser.add_argument("--folds", nargs="+", type=int)
parser.add_argument("--epochs", default=12, type=int)
parser.add_argument("--workers", default=8, type=int)
parser.add_argument("--patience", default=15, type=int)
parser.add_argument("--resume", default=False, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--quick", default=False, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--fp16", default=True, type=lambda x: (str(x).lower() == "true"))
parser.add_argument("--dgx", default=False, type=lambda x: (str(x).lower() == "true"))
args = parser.parse_args()

print('args:',args)
if __name__ == "__main__":
    rsna = RSNAConfig()
    rsna.update(args.net, args.fp16, args.dgx)
    print('cfg:',rsna.conf)

    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('logs'):
        os.makedirs('logs')

    transforms_train, transforms_test = getTransforms(rsna.conf.size)

    df = pd.read_csv(rsna.conf.new_trainset)

    for fold in args.folds:
        print('*'*40 + ' FOLD {} '.format(fold) + '*'*40)
        valid_df = df.loc[df['fold'] == fold]
        train_df = df.loc[~df.index.isin(valid_df.index)]
        train_df = train_df.sample(frac=1).reset_index(drop=True)

        if args.quick:
            train_df = train_df.sample(400)
            valid_df = valid_df.sample(100)

        trainset = RSNADataset(df = train_df, root_dir = rsna.conf.stage1_train_dir, transform = transforms_train, datatype = 'train')
        train_loader = DataLoader(trainset, batch_size = rsna.conf.batch_size, shuffle = True, num_workers = args.workers)

        valset = RSNADataset(df = valid_df, root_dir = rsna.conf.stage1_train_dir, transform = transforms_test, datatype = 'train')
        valid_loader = DataLoader(valset, batch_size = rsna.conf.batch_size, shuffle = False, num_workers = args.workers)

        model = make_model(model_name = rsna.conf.network, num_classes = rsna.conf.num_classes)
        model = model.cuda()
        train_criterion = nn.BCEWithLogitsLoss()

        weight_tensor = torch.FloatTensor([2., 1., 1., 1., 1., 1.]).cuda()
        val_criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)

        MODEL_CHECKPOINT = 'checkpoints/{}_fold{}.pt'.format(rsna.conf.network, fold)
        TRAINING_LOG = 'logs/{}_fold{}.csv'.format(rsna.conf.network, fold)

        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        if args.fp16:
            model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")

        scheduler = CyclicLRWithRestarts(
            optimizer = optimizer,
            batch_size = rsna.conf.batch_size,
            epoch_size = len(train_loader.dataset),
            restart_period=5,
            t_mult=1.2,
            policy="cosine")

        if args.resume:
            checkpoint = torch.load(MODEL_CHECKPOINT)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            if args.fp16:
                amp.load_state_dict(checkpoint['amp'])
            valid_loss_min = checkpoint['valid_loss_min']
            epoch_init = checkpoint['epoch_init']
        else:
            valid_loss_min = np.Inf
            epoch_init = 0

        if epoch_init == 0 and os.path.isfile(TRAINING_LOG):
            os.remove(TRAINING_LOG)

        pat = 0
        for epoch in range(epoch_init, args.epochs, 1):
            epoch_start_time = time.time()

            train_loss = []
            model.train()

            loop = tqdm(train_loader)
            for inputs, labels in loop:
                inputs = inputs.cuda()
                labels = labels.cuda()

                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels, 0.5, True)
                inputs, labels_a, labels_b = map(Variable, (inputs, labels_a, labels_b))

                with torch.set_grad_enabled(True):
                    outputs = model(inputs)
                    loss = mixup_criterion(train_criterion, outputs, labels_a, labels_b, lam)

                    optimizer.zero_grad()
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    optimizer.step()
                    scheduler.batch_step()

                train_loss.append(loss.item())

                loop.set_description('Epoch {:2d}/{:2d}'.format(epoch, args.epochs-1))
                loop.set_postfix(loss=np.mean(train_loss))
            train_loss = np.mean(train_loss)

            scheduler.step()

            model.eval()

            valid_loss = 0.0
            for inputs, labels in tqdm(valid_loader):
                inputs = inputs.cuda()
                labels = labels.cuda()

                with torch.set_grad_enabled(False):
                    outputs = model(inputs)
                    if len(outputs.size()) == 1:
                        outputs = torch.unsqueeze(outputs, 0)
                    loss = val_criterion(outputs, labels)
                valid_loss += loss.item() * inputs.size(0)

            valid_loss = valid_loss / len(valid_loader.dataset)
            valid_loss = valid_loss*6.0/7.0

            epoch_elapsed_time = time.time() - epoch_start_time
            hours = epoch_elapsed_time//3600
            minutes = (epoch_elapsed_time%3600)//60
            secs = epoch_elapsed_time - hours*3600 - minutes*60

            print('Time: {:2.0f}h {:2.0f}m {:2.0f}s | Train loss: {:.5f} | Val loss: {}'.format(hours, minutes, secs, train_loss, valid_loss))
            log_file = open(TRAINING_LOG, 'a')
            log_file.write('Epoch: {:2d}/{:2d} | Time: {:2.0f}h {:2.0f}m {:2.0f}s | Train loss: {:.5f} | Val log loss: {:.5f}\n'.format(
                            epoch, args.epochs-1, hours, minutes, secs, train_loss, valid_loss))
            log_file.close()

            if valid_loss <= valid_loss_min:
                print('Valid log loss improved from {:.5f} to {:.5f} saving model to {}'.format(valid_loss_min, valid_loss, MODEL_CHECKPOINT))
                valid_loss_min = valid_loss
                pat = 0
                if args.fp16:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'amp': amp.state_dict(),
                        'valid_loss_min': valid_loss_min,
                        'epoch_init': epoch+1,
                    }, MODEL_CHECKPOINT)
                else:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'valid_loss_min': valid_loss_min,
                        'epoch_init': epoch+1,
                    }, MODEL_CHECKPOINT)
            else:
                pat += 1

            if epoch in rsna.conf.snapshot_epochs:
                if args.fp16:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'amp': amp.state_dict(),
                        'valid_loss_min': valid_loss_min,
                        'epoch_init': epoch+1,
                    }, MODEL_CHECKPOINT.replace('.pt', '_epoch{}.pt'.format(epoch)))
                else:
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),
                        'valid_loss_min': valid_loss_min,
                        'epoch_init': epoch+1,
                    }, MODEL_CHECKPOINT.replace('.pt', '_epoch{}.pt'.format(epoch)))

            if pat == args.patience:
                break
