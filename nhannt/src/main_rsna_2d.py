from rsna_dataset import RSNAHemorrhageDS2d, EasySampler
from utils import *
from model import GenericEfficientNet
from lr_scheduler import LR_Scheduler
from config import configs2d
import argparse
import ast
import numpy as np
import os
import pandas as pd
import random
import time
from tqdm import tqdm

import apex
from apex import amp

import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader


parser = argparse.ArgumentParser(
    description='PyTorch Training')
parser.add_argument('--start-epoch', default=0, type=int,
                    metavar='N', help='start epoch')
parser.add_argument('-e', '--eval', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('-et', '--eval-test', action='store_true',
                    help='run inference on test set')
parser.add_argument('-ft', '--finetune', action='store_true',
                    help='whether to finetune')
parser.add_argument('--dtype', default='float16', type=str,
                    help='full/mixed precision training')
parser.add_argument('--opt_level', default="O2", type=str)
parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                    help="Modify config options using the command-line")


def train(cfg, train_loader, model, criterion,
          optimizer, scheduler, epoch):
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    for i, (image, target) in enumerate(tbar):
        image = image.cuda()
        target = target.cuda()
        # Main branch & auxiliary
        if cfg["CUTMIX"] and np.random.uniform() < cfg["P_AUGMENT"]:
            mixed_x, y_a, y_b, lam = cutmix_data(image, target)
            output, aux_output0, aux_output1 = model(mixed_x)
            loss = mixup_criterion(criterion, output, y_a, y_b, lam) + cfg["AUX_W"] * (mixup_criterion(
                criterion, aux_output0, y_a, y_b, lam) + mixup_criterion(
                    criterion, aux_output1, y_a, y_b, lam))
        else:
            output, aux_output0, aux_output1 = model(image)
            loss = criterion(output, target) + cfg["AUX_W"] * (criterion(aux_output0, target) \
                + criterion(aux_output1, target))
                
        # gradient accumulation
        loss = loss / cfg['GD_STEPS']
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        if (i + 1) % cfg['GD_STEPS'] == 0:
            scheduler(optimizer, i, epoch)
            optimizer.step()
            optimizer.zero_grad()
        
        # record loss
        losses.update(loss.item() * cfg['GD_STEPS'], image.size(0))
        tbar.set_description("Train loss: %.5f, learning rate: %.6f"
                             % (losses.avg,
                                optimizer.param_groups[-1]['lr']))


def validate(cfg, valid_loader, model, valid_criterion):
    # switch to evaluate mode
    model.eval()

    logit_tensor = []
    loss_array = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            output = model(image)

            # compute loss with logits
            loss = valid_criterion(output, target)
            loss[torch.isnan(loss)] = 0.
            
            loss_array.append(loss.cpu().numpy())
            logit_tensor.append(output.cpu())

    logit_tensor = torch.cat(logit_tensor, 0)
    torch.save(logit_tensor, os.path.join(cfg['OUTPUT_DIR'], f"val_{cfg['SESS_NAME']}.pth"))
    
    loss_array = np.concatenate(loss_array, 0)
    val_loss = loss_array.mean()
    any_loss = loss_array[:, 0].mean()
    intraparenchymal_loss = loss_array[:, 1].mean()
    intraventricular_loss = loss_array[:, 2].mean()
    subarachnoid_loss = loss_array[:, 3].mean()
    subdural_loss = loss_array[:, 4].mean()
    epidural_loss = loss_array[:, 5].mean()

    logger.info("Validation loss: {:.5f} - any: {:.3f} - intraparenchymal: {:.3f} - intraventricular: {:.3f} - subarachnoid: {:.3f} - subdural: {:.3f} - epidural: {:.3f}\n".format(
        val_loss, any_loss, intraparenchymal_loss, intraventricular_loss, subarachnoid_loss, subdural_loss, epidural_loss))
    return val_loss


def test(cfg, test_loader, model):
    # switch to evaluate mode
    model.eval()

    ids = []
    probs = []

    tbar = tqdm(test_loader)
    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            image = image.cuda()
            output = model(image)
            output = torch.sigmoid(output)
            probs.append(output.cpu().numpy())
            ids += id_code

    probs = np.concatenate(probs, 0)
    # np.save(
    #     os.path.join(
    #         cfg["OUTPUT_DIR"],
    #         f"test_{cfg['SESS_NAME']}.npy"),
    #     probs)
    submit = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    submit.columns = [
        "image",
        "any",
        "intraparenchymal",
        "intraventricular",
        "subarachnoid",
        "subdural",
        "epidural"]
    return submit


def main(cfg):
    global best_loss
    best_loss = 100.

    # create dataset
    train_ds = RSNAHemorrhageDS2d(cfg, mode="train")
    valid_ds = RSNAHemorrhageDS2d(cfg, mode="valid")
    test_ds = RSNAHemorrhageDS2d(cfg, mode="test")

    # create model
    extra_model_args = {"attention": cfg["ATTENTION"]}
    model = GenericEfficientNet(cfg["MODEL_NAME"], input_channels=cfg["NUM_INP_CHAN"],
                                num_classes=6, **extra_model_args)

    # define loss function (criterion) and optimizer
    criterion = nn.BCEWithLogitsLoss(weight=cfg["BCE_W"])
    valid_criterion = nn.BCEWithLogitsLoss(weight=cfg["BCE_W"], reduction='none')

    optimizer = make_optimizer(cfg, model)

    if cfg["CUDA"]:
        model = model.cuda()
        criterion = criterion.cuda()
        valid_criterion = valid_criterion.cuda()

    if args.dtype == 'float16':
        if args.opt_level == "O1":
            keep_batchnorm_fp32 = None
        else:
            keep_batchnorm_fp32 = True
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          opt_level=args.opt_level,
                                          keep_batchnorm_fp32=keep_batchnorm_fp32)

    start_epoch = 0
    # optionally resume from a checkpoint
    if cfg["RESUME"]:
        if os.path.isfile(cfg["RESUME"]):
            logger.info("=> Loading checkpoint '{}'".format(cfg["RESUME"]))
            checkpoint = torch.load(cfg["RESUME"], "cpu")
            load_state_dict(checkpoint.pop('state_dict'), model)
            if not args.finetune:
                start_epoch = checkpoint['epoch']
                optimizer.load_state_dict(checkpoint.pop('optimizer'))
                best_loss = checkpoint['best_loss']
            logger.info("=> Loaded checkpoint '{}' (epoch {})"
                  .format(cfg["RESUME"], checkpoint['epoch']))
        else:
            logger.info("=> No checkpoint found at '{}'".format(cfg["RESUME"]))

    if cfg['MULTI_GPU']:
        model = nn.DataParallel(model)

    # Create sampler
    easy_img_ids = pd.read_csv(
        cfg["DATA_DIR"] +
        "easy_train.csv")["image"].values
    easy_samples = train_ds.source[train_ds.source["image"].isin(easy_img_ids)]
    hard_samples = train_ds.source[~train_ds.source["image"].isin(
        easy_img_ids)]
    train_sampler = EasySampler(
        hard_samples.index,
        easy_samples.index,
        ratio=cfg["RATIO"])

    # Data loading code
    train_loader = DataLoader(train_ds, cfg["BATCH_SIZE"], pin_memory=False,
                              sampler=train_sampler,
                              drop_last=True, num_workers=cfg['NUM_WORKERS'])
    valid_loader = DataLoader(valid_ds, cfg["BATCH_SIZE"], pin_memory=False,
                              shuffle=False, drop_last=False, num_workers=cfg['NUM_WORKERS'])
    test_loader = DataLoader(test_ds, cfg["BATCH_SIZE"], pin_memory=False,
                             shuffle=False, drop_last=False, num_workers=cfg['NUM_WORKERS'])

    scheduler = LR_Scheduler("cos", cfg["BASE_LR"], cfg["EPOCHS"],
                             iters_per_epoch=len(train_loader),
                             warmup_epochs=cfg["WARMUP_EPOCHS"])
    logger.info("Using {} lr scheduler\n".format(scheduler.mode))

    if args.eval:
        validate(cfg, valid_loader, model, valid_criterion)
        return

    if args.eval_test:
        if not os.path.exists(cfg["OUTPUT_DIR"]):
            os.makedirs(cfg["OUTPUT_DIR"])
        submit_df = test(cfg, test_loader, model)
        submit_df.to_csv(
            os.path.join(cfg["OUTPUT_DIR"], "test_" + cfg["SESS_NAME"] + '.csv'),
            index=False)
        return

    for epoch in range(start_epoch, cfg["EPOCHS"]):
        logger.info("Epoch {}\n".format(str(epoch + 1)))
        random.seed(epoch)
        torch.manual_seed(epoch)
        # train for one epoch
        train(
            cfg,
            train_loader,
            model,
            criterion,
            optimizer,
            scheduler,
            epoch)
        # evaluate
        loss = validate(cfg, valid_loader, model, valid_criterion)
        # remember best loss and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)
        if cfg["MULTI_GPU"]:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg["MODEL_NAME"],
                'state_dict': model.module.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, root=cfg['MODELS_DIR'], filename=f"{cfg['SESS_NAME']}.pth")
        else:
            save_checkpoint({
                'epoch': epoch + 1,
                'arch': cfg["MODEL_NAME"],
                'state_dict': model.state_dict(),
                'best_loss': best_loss,
                'optimizer': optimizer.state_dict(),
            }, is_best, root=cfg['MODELS_DIR'], filename=f"{cfg['SESS_NAME']}.pth")


if __name__ == '__main__':
    cfg = configs2d
    args = parser.parse_args()
    opts = {}
    for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
        opts[full_key] = ast.literal_eval(v)
    cfg.update(opts)

    # create logger
    global logger
    logger = setup_logger("2D Training", cfg["LOGS_DIR"], 
        cfg["LOCAL_RANK"], cfg["SESS_NAME"] + ".txt")
    logger.info("{}\n".format(args))
    logger.info("{}\n".format(cfg))
    
    main(cfg)