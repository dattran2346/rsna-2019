import os
import sys
import argparse
import logging
import random

import apex
from apex import amp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Subset


from config import get_cfg_defaults
from models import train_loop, valid_model, test_model, get_model, \
                   OctResNet50, DenseNet
from datasets import RSNAHemorrhageDS, EasySampler, RSNAHemorrhageDS_RNN, DebugSampler
from lr_scheduler import LR_Scheduler
from cyclic_scheduler import CyclicLRWithRestarts

def parse_args():

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="",
            help="config yaml path")
    parser.add_argument("--load", type=str, default="",
            help="path to model weight")
    parser.add_argument("-ft", "--finetune", action="store_true",
        help="path to model weight")
    parser.add_argument("-m", "--mode", type=str, default="train",
            help="model runing mode (train/valid/test)")
    parser.add_argument("--valid", action="store_true",
            help="enable evaluation mode for validation")
    parser.add_argument("--test", action="store_true",
            help="enable evaluation mode for testset")
    parser.add_argument("--tta", action="store_true",
            help="enable tta infer")
    parser.add_argument("--smooth", action="store_true",
            help="enable smooth infer")

    parser.add_argument("-d", "--debug", action="store_true",
            help="enable debug mode for test")

    args = parser.parse_args()
    if args.valid:
        args.mode = "valid"
    elif args.test:
        args.mode = "test"

    return args


def setup_logging(args, cfg):

    if not os.path.isdir(cfg.DIRS.LOGS):
        os.mkdir(cfg.DIRS.LOGS)
    
    head = '{asctime}:{levelname}: {message}'
    handlers = [logging.StreamHandler(sys.stderr)]
    handlers.append(logging.FileHandler(
        os.path.join(cfg.DIRS.LOGS, f'{cfg.EXP}.{args.mode}.log'), 
        mode='a'))
    logging.basicConfig(level=logging.DEBUG, format=head, style='{', handlers=handlers)
    logging.info(f'Start with config {cfg}')
    logging.info(f'Command arguments {args}')


def setup_determinism(cfg):

    seed = cfg.SYSTEM.SEED
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def main(args, cfg):

    logging.info(f"=========> {cfg.EXP} <=========")

    # Declare variables
    start_epoch = 0
    best_metric = 100.

    # Create model
    model = get_model(cfg)

    # Define Loss and Optimizer
    train_criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(cfg.CONST.BCE_W))
    valid_criterion = nn.BCEWithLogitsLoss(weight=torch.FloatTensor(cfg.CONST.BCE_W), reduction='none')
    if cfg.TRAIN.NUM_CLASSES == 1:
        train_criterion = nn.BCEWithLogitsLoss()
        valid_criterion = nn.BCEWithLogitsLoss(reduction='none')
    if args.valid:
        valid_criterion = nn.BCELoss(weight=torch.FloatTensor(cfg.CONST.BCE_W), reduction='none')
    optimizer = optim.AdamW(params=model.parameters(), 
                            lr=cfg.OPT.BASE_LR, 
                            weight_decay=cfg.OPT.WEIGHT_DECAY)

    # CUDA & Mixed Precision
    if cfg.SYSTEM.CUDA:
        model = model.cuda()
        train_criterion = train_criterion.cuda()
        valid_criterion = valid_criterion.cuda()
    
    if cfg.SYSTEM.FP16:
        model, optimizer = amp.initialize(models=model, optimizers=optimizer, 
                                          opt_level=cfg.SYSTEM.OPT_L, 
                                          keep_batchnorm_fp32=(True if cfg.SYSTEM.OPT_L == "O2" else None))

    # Load checkpoint
    if args.load != "":
        if os.path.isfile(args.load):
            print(f"=> loading checkpoint {args.load}")
            ckpt = torch.load(args.load, "cpu")
            if cfg.TRAIN.MODEL == "octave-resnet50-hybrid":
                model_dict = model.state_dict()
                model_dict.update(ckpt.pop('state_dict'))
                model.load_state_dict(model_dict)
            else:
                model.load_state_dict(ckpt.pop('state_dict'))
            if not args.finetune:
                print("resuming optimizer ...")
                optimizer.load_state_dict(ckpt.pop('optimizer'))
                start_epoch, best_metric = ckpt['epoch'], ckpt['best_metric']
            logging.info(f"=> loaded checkpoint '{args.load}' (epoch {ckpt['epoch']}, best_metric: {ckpt['best_metric']})")
        else:
            logging.info(f"=> no checkpoint found at '{args.load}'")

    if cfg.SYSTEM.MULTI_GPU:
        model = nn.DataParallel(model)

    # Load data
    if cfg.TRAIN.CRNN:
        DataSet = RSNAHemorrhageDS_RNN
    else:
        DataSet = RSNAHemorrhageDS

    train_ds = DataSet(cfg, mode="train")
    valid_ds = DataSet(cfg, mode="valid")
    test_ds = DataSet(cfg, mode="test")
    
    # Dataloader

    valid_bs = 1 if cfg.TRAIN.CRNN else cfg.TRAIN.BATCH_SIZE

    if cfg.DEBUG:
        train_ds = Subset(train_ds, np.random.choice(np.arange(len(train_ds)), 256))
        valid_ds = Subset(valid_ds, np.random.choice(np.arange(len(valid_ds)), 10))
        test_ds = Subset(test_ds, np.random.choice(np.arange(len(test_ds)), 10))
        train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, 
                                shuffle=True, drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
        valid_loader = DataLoader(valid_ds, valid_bs, pin_memory=False,
                                 shuffle=False, drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS) 
        test_loader = DataLoader(test_ds, valid_bs, pin_memory=False,
                                 shuffle=False, drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS) 
    else:
        if cfg.TRAIN.CRNN:

            train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, 
                                    shuffle=True,
                                    drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
            valid_loader = DataLoader(valid_ds, valid_bs, pin_memory=False, shuffle=False, 
                                drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)
            test_loader = DataLoader(test_ds, valid_bs, pin_memory=False, shuffle=False, 
                             drop_last=False, num_workers=cfg.SYSTEM.NUM_WORKERS)

        else:
            easy_img_ids = pd.read_csv(os.path.join(cfg.DIRS.DATA + "easy_train.csv"))["image"].values
            easy_samples = train_ds.source[train_ds.source["image"].isin(easy_img_ids)]
            hard_samples = train_ds.source[~train_ds.source["image"].isin(easy_img_ids)]
            train_sampler = EasySampler(hard_samples.index, easy_samples.index, ratio=1.0)

            train_loader = DataLoader(train_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, 
                                    shuffle=False, sampler=train_sampler,
                                    drop_last=False, num_workers=int(cfg.SYSTEM.NUM_WORKERS))

            valid_loader = DataLoader(valid_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, shuffle=False, 
                                drop_last=False, num_workers=int(cfg.SYSTEM.NUM_WORKERS))

            test_loader = DataLoader(test_ds, cfg.TRAIN.BATCH_SIZE, pin_memory=False, shuffle=False, 
                             drop_last=False, num_workers=int(cfg.SYSTEM.NUM_WORKERS))


    scheduler = LR_Scheduler("cos", cfg.OPT.BASE_LR, cfg.TRAIN.EPOCHS,\
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg.OPT.WARMUP_EPOCHS)


    if args.mode == "train":
        train_loop(logging.info, cfg, model, \
                train_loader, train_criterion, valid_loader, valid_criterion, \
                optimizer, scheduler, start_epoch, best_metric)
    elif args.mode == "valid":
        valid_model(logging.info, cfg, model, valid_loader, valid_criterion, smooth_valid=cfg.INFER.SMOOTH, tta=cfg.INFER.TTA)
    else:
        test_model(logging.info, cfg, model, test_loader, smooth=cfg.INFER.SMOOTH, tta=cfg.INFER.TTA)

if __name__ == "__main__":

    args = parse_args()
    cfg = get_cfg_defaults()
    if args.config != "":
        cfg.merge_from_file(args.config)
    
    if args.mode != "train":
        cfg.merge_from_list(['INFER.TTA', args.tta, 'INFER.SMOOTH', args.smooth])

    if args.debug:
        opts = ["DEBUG", True, "TRAIN.EPOCHS", 2]
        cfg.merge_from_list(opts)
    cfg.freeze()
    setup_logging(args, cfg)
    setup_determinism(cfg)
    main(args, cfg)