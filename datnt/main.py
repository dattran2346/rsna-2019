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

from dataset import RSNADataset
from augmenter import ValAugmenter, HarderAugmenter, MegaAugmenter
from torch.utils.data import DataLoader
from torch import optim, nn
from torch.autograd import Variable
from sklearn.metrics import log_loss
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR
from my_scheduler import WarmUpLR
from tqdm import tqdm
from apex import amp

def main(cfg):
    cfg.classname = cfg.classname.split(",")
    cfg.lr = cfg.lr * cfg.gdstep

    if not os.path.isdir(cfg.outdir):
        os.makedirs(cfg.outdir)
    
    terminal_logger = os.path.join(cfg.outdir, 'terminal.txt')
    f = open(terminal_logger, 'w')
    f.close()
    print(cfg, file=open(terminal_logger, 'a'))
    print(cfg)
    
    model = backbone.make_model(cfg.modelname, cfg.classname)
    if cfg.resume != 'imagenet':
        old_checkpoint = torch.load(cfg.resume)
        model.load_state_dict(old_checkpoint, strict=False)
        
    train_dataset = RSNADataset(dataset_csv_file=os.path.join(cfg.csvdir, f'train_fold{cfg.fold}.csv'),
                                class_names=cfg.classname,
                                source_image_dir=cfg.datadir,
                                augmenter=MegaAugmenter(cfg.imgsize))
    validation_dataset = RSNADataset(dataset_csv_file=os.path.join(cfg.csvdir, f'val_fold{cfg.fold}.csv'),
                                     class_names=cfg.classname,
                                     source_image_dir=cfg.datadir,
                                     augmenter=ValAugmenter(cfg.imgsize))
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.batchsize, shuffle=True, num_workers=cfg.workers)
    validation_dataloader = DataLoader(validation_dataset, batch_size=cfg.batchsize, shuffle=False, num_workers=cfg.workers)

    assert cfg.optim in ['sgd', 'adam']
    splited_params = regularizer.split_weights(model)
    if cfg.optim == 'sgd':
        optimizer = optim.SGD(splited_params, lr=cfg.lr, momentum=0.9, weight_decay=0.01)
    elif cfg.optim == 'adam':
        optimizer = optim.AdamW(splited_params, lr=cfg.lr, weight_decay=0.01)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O2", keep_batchnorm_fp32=True, loss_scale="dynamic")
    
    assert 0 < cfg.topk
    assert cfg.topk <= 1
    if cfg.topk == 1:
        if cfg.finetune:
            weight_tensor = torch.FloatTensor([1., 0., 0., 0., 0., 0.]).cuda()
        else:
            weight_tensor = torch.FloatTensor([2., 1., 1., 1., 1., 1.]).cuda()
        criterion = nn.BCEWithLogitsLoss(weight=weight_tensor)
    else:
        criterion = nn.BCEWithLogitsLoss(reduction='none')
        criterion = losses.OHEM_Loss(top_k=0.7, loss_func=criterion)

    cosine_scheduler = CosineAnnealingLR(optimizer, T_max = cfg.epoch*len(train_dataloader)/cfg.gdstep, eta_min=cfg.lr/100)
    warmup_scheduler = WarmUpLR(optimizer, len(train_dataloader)/cfg.gdstep)

    best_scr = np.inf
    if cfg.finetune:
        best_scr = validate(model, validation_dataloader, terminal_logger, cfg)
    for e in range(cfg.epoch):
        print(f'Epoch {e+1}/{cfg.epoch}', file=open(terminal_logger, 'a'))
        print(f'Epoch {e+1}/{cfg.epoch}')
        if e == 0:
            train(model, train_dataloader, criterion, optimizer, warmup_scheduler, terminal_logger, cfg)
        else:
            train(model, train_dataloader, criterion, optimizer, cosine_scheduler, terminal_logger, cfg)
        scr = validate(model, validation_dataloader, terminal_logger, cfg)

        model_name = f'{cfg.outdir}/{cfg.modelname}.pth'

        if scr < best_scr:
            best_scr = scr
            logger = open(os.path.join(cfg.outdir, 'logger.txt'), 'a')
            logger.writelines(str(best_scr)+'\t'+str(e+1)+'\n')
            logger.close()
            torch.save(model.state_dict(), model_name)

def train(model, dataloader, criterion, optimizer, scheduler, terminal_logger, cfg):

    model.train()

    # y_true = []
    # y_pred = []
    losses = []
    tbar = tqdm(dataloader)

    for i, (input, target) in enumerate(tbar):
        input = input.cuda()
        target = target.cuda()
        if np.random.uniform() < cfg.pcut and cfg.cutmix:
            mixed_x, y_a, y_b, lam = regularizer.cutmix_data(input, target)
            output = model(mixed_x)
            if '_aux' not in cfg.modelname:
                loss = regularizer.mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                loss0 = regularizer.mixup_criterion(criterion, output[0], y_a, y_b, lam)
                loss1 = regularizer.mixup_criterion(criterion, output[1], y_a, y_b, lam)
                loss = loss0 * 0.7 + loss1 * 0.3
        else:
            input_var = torch.autograd.Variable(input, requires_grad=True)
            target_var = torch.autograd.Variable(target)
            output = model(input_var)
            if '_aux' not in cfg.modelname:
                loss = criterion(output, target_var)
            else:
                loss0 = criterion(output[0], target_var)
                loss1 = criterion(output[1], target_var)
                loss = loss0 * 0.7 + loss1 * 0.3

        loss /= cfg.gdstep
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()

        if (i+1) % cfg.gdstep == 0:
            scheduler.step()
            optimizer.step()
            optimizer.zero_grad()
        loss = loss.detach().cpu().numpy()
        losses.append(loss*cfg.gdstep)

        # y_true += [target.detach().cpu().numpy()]
        # if not aux:
        #     y_pred += [output.detach().cpu().numpy()]
        # else:
        #     y_pred += [output[0].detach().cpu().numpy()]

        tbar.set_description(f'Loss: {loss*cfg.gdstep:.4f} ({np.mean(losses):.4f}), lr={optimizer.param_groups[0]["lr"]:.4e}')

def validate(model, dataloader, terminal_logger, cfg):
    model.eval()

    y_true = []
    y_pred = []
    tbar = tqdm(dataloader)
    for _, (x, y_t) in enumerate(tbar):
        with torch.no_grad():
            y_t = y_t.cuda()
            x = Variable(x).cuda()
            if '_aux' not in cfg.modelname:
                y_p = torch.sigmoid(model(x))
            else:
                y_p = torch.sigmoid(model(x)[0]) * 0.8 + torch.sigmoid(model(x)[1]) * 0.2

            y_true += [y_t.detach().cpu().numpy()]
            y_pred += [y_p.detach().cpu().numpy()]

    ll = []
    y_true = np.vstack(y_true)
    y_pred = np.vstack(y_pred)
    for i in range(len(cfg.classname)):
        yti = y_true[:, i]
        ypi = np.clip(y_pred[:, i], 1e-7, 1 - 1e-7)
        score = log_loss(yti, ypi)
        if i == 0:
            score *= 2
        ll.append(score)
    if cfg.finetune:
        ll = ll[0]/2.

    print(f'Validation mean LL: {np.mean(ll):.6f}',  file=open(terminal_logger, 'a'))
    print(f'Validation mean LL: {np.mean(ll):.6f}')
    return np.mean(ll)
                             
def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True                             

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', type=str)
    parser.add_argument('--modelname', type=str)
    parser.add_argument('--csvdir', type=str)
    parser.add_argument('--fold', type=str)
    parser.add_argument('--finetune', type=bool, default=False)
    parser.add_argument('--gdstep', type=int, default=1)
    parser.add_argument('--datadir', type=str, default='rsna-crop-384')
    parser.add_argument('--classname', type=str, default='any,epidural,intraparenchymal,intraventricular,subarachnoid,subdural')
    parser.add_argument('--epoch', type=int, default=25)
    parser.add_argument('--batchsize', type=int, default=64)
    parser.add_argument('--imgsize', type=int, default=384)
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--topk', type=float, default=1)
    parser.add_argument('--optim', type=str, default='adam')
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--resume', type=str, default='imagenet')
    parser.add_argument('--showmodel', type=bool, default=False)
    parser.add_argument('--cutmix', type=bool, default=False)
    parser.add_argument('--pcut', type=float, default=0.5)
                             
    cfg = parser.parse_args()
    
    seed_everything()
                             
    main(cfg)
