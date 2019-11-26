import argparse
import ast
import numpy as np
import os 
import pandas as pd
import random
import time
from tqdm import tqdm
import torch
torch.multiprocessing.set_sharing_strategy("file_system")
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
from torch.utils.data import DataLoader
import apex
from apex import amp
from configs.rsna_3d import configs
from contrib import KnowledgeDistillationLoss, WarmupCyclicalLR
from data import RSNAHemorrhageDS3d
from model import GenericEfficientNet3d, ResNet3d
from utils import AverageMeter
from utils import setup_logger
from utils import cutmix_data, mixup_data, mixup_criterion
from utils import make_optimizer
from utils import test_collate_fn, tta
from utils import load_state_dict, save_checkpoint


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


def train(cfg, train_loader, model, criterion, kd_criterion,
          optimizer, scheduler, epoch):
    """
    Helper function to train.
    """
    losses = AverageMeter()
    model.train()
    tbar = tqdm(train_loader)

    for i, (image, target) in enumerate(tbar):
        image = image.cuda()
        target = target.cuda()
        bsize, seq_len, c, h, w = image.size()
        # image = image.view(bsize * seq_len, c, h, w)
        # target = target.view(-1, target.size(-1))
        
        data_aug = cfg["CUTMIX"] or cfg["MIXUP"]
        if np.random.uniform() < cfg["P_AUGMENT"] and data_aug:
        #     if cfg["CUTMIX"]:
        #         mixed_x, y_a, y_b, lam = cutmix_data(image, target)
        #     elif cfg["MIXUP"]:
        #         mixed_x, y_a, y_b, lam = mixup_data(image, target)
            mixed_x = []
            y_a = []
            y_b = []
            lam = []
            for st_image, st_target in zip(image, target):
                mixed_st_image, st_y_a, st_y_b, st_lam = cutmix_data(st_image, st_target)
                mixed_x.append(mixed_st_image)
                y_a.append(st_y_a)
                y_b.append(st_y_b)
                lam.append(torch.FloatTensor([st_lam] * seq_len))
            mixed_x = torch.stack(mixed_x)
            y_a = torch.stack(y_a)
            y_b = torch.stack(y_b)
            lam = torch.cat(lam, 0).unsqueeze(1).cuda()
            mixed_x = mixed_x.view(bsize * seq_len, c, h, w)
            y_a = y_a.view(-1, target.size(-1))
            y_b = y_b.view(-1, target.size(-1))

            output, aux_output0, aux_output1 = model(mixed_x, seq_len)
            main_loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            if cfg["USE_KD"]:
                aux_loss = cfg["ALPHA"] * (mixup_criterion(criterion, aux_output0, y_a, y_b, lam) + mixup_criterion(
                    criterion, aux_output1, y_a, y_b, lam)) + (1. - cfg["ALPHA"]) * (kd_criterion(aux_output0, output) + kd_criterion(
                    aux_output1, output))
            else:
                aux_loss = mixup_criterion(criterion, aux_output0, y_a, y_b, lam) + mixup_criterion(
                    criterion, aux_output1, y_a, y_b, lam)
        else:
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))
            output, aux_output0, aux_output1 = model(image, seq_len)
            main_loss = criterion(output, target)
            if cfg["USE_KD"]:
                aux_loss = cfg["ALPHA"] * (criterion(aux_output0, target) + criterion(
                    aux_output1, target)) + (1. - cfg["ALPHA"]) * (kd_criterion(aux_output0, output) + kd_criterion(
                    aux_output1, output))
            else:
                aux_loss = criterion(aux_output0, target) + criterion(aux_output1, target)
        loss = main_loss + cfg["AUX_W"] * aux_loss
        loss = loss.mean()
        
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

    prob_tensor = []
    loss_array = []
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            target = target.view(-1, target.size(-1))            
            if cfg["TTA"]:
                output = tta(model, image, seq_len)
            else:
                output = model(image, seq_len)
            loss = valid_criterion(output, target)
            loss_array.append(loss.cpu().numpy())
            prob_tensor.append(torch.sigmoid(output).cpu())
    
    prob_tensor = torch.cat(prob_tensor, 0)
    # record loss
    loss_array = np.concatenate(loss_array, 0)
    val_loss = loss_array.mean()
    any_loss = loss_array[:, 0].mean()
    intraparenchymal_loss = loss_array[:, 1].mean()
    intraventricular_loss = loss_array[:, 2].mean()
    subarachnoid_loss = loss_array[:, 3].mean()
    subdural_loss = loss_array[:, 4].mean()
    epidural_loss = loss_array[:, 5].mean()

    logger.info("Validation loss: {:.5f} - any: {:.3f} - intraparenchymal: {:.3f} - intraventricular: {:.3f} - subarachnoid: {:.3f} - subdural: {:.3f} - epidural: {:.3f}\n".format(
        val_loss, any_loss, 
        intraparenchymal_loss, intraventricular_loss, 
        subarachnoid_loss, subdural_loss, epidural_loss))
    return val_loss, prob_tensor


def test(cfg, test_loader, model):
    # switch to evaluate mode
    model.eval()
    ids = []
    probs = []
    tbar = tqdm(test_loader)

    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            image = image.cuda()
            bsize, seq_len, c, h, w = image.size()
            image = image.view(bsize * seq_len, c, h, w)
            if cfg["TTA"]:
                output = tta(model, image, seq_len)
            else:
                output = model(image, seq_len)
            output = torch.sigmoid(output)
            probs.append(output.cpu().numpy())
            ids += id_code

    probs = np.concatenate(probs, 0)
    submit = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    submit.columns = ["image", "any",
                      "intraparenchymal", "intraventricular",
                      "subarachnoid", "subdural", "epidural"]
    return submit


def main(cfg):
    global best_loss
    best_loss = 100.

    # make dirs
    for dirs in [cfg["MODELS_DIR"], cfg["OUTPUT_DIR"], cfg["LOGS_DIR"]]:
        if not os.path.exists(dirs):
            os.makedirs(dirs)

    # create dataset
    train_ds = RSNAHemorrhageDS3d(cfg, mode="train")
    valid_ds = RSNAHemorrhageDS3d(cfg, mode="valid")
    test_ds = RSNAHemorrhageDS3d(cfg, mode="test")

    # create model
    extra_model_args = {"attention": cfg["ATTENTION"],
        "dropout": cfg["DROPOUT"],
        "num_layers": cfg["NUM_LAYERS"],
        "recur_type": cfg["RECUR_TYPE"],
        "num_heads": cfg["NUM_HEADS"],
        "dim_ffw": cfg["DIM_FFW"]}
    if cfg["MODEL_NAME"].startswith("tf_efficient"):
        model = GenericEfficientNet3d(cfg["MODEL_NAME"], input_channels=cfg["NUM_INP_CHAN"],
                                      num_classes=cfg["NUM_CLASSES"], **extra_model_args)
    elif "res" in cfg["MODEL_NAME"]:
        model = ResNet3d(cfg["MODEL_NAME"], input_channels=cfg["NUM_INP_CHAN"],
                         num_classes=cfg["NUM_CLASSES"], **extra_model_args)
    # print(model)

    # define loss function & optimizer
    class_weight = torch.FloatTensor(cfg["BCE_W"])
    # criterion = nn.BCEWithLogitsLoss(weight=class_weight)
    criterion = nn.BCEWithLogitsLoss(weight=class_weight, reduction='none')
    kd_criterion = KnowledgeDistillationLoss(temperature=cfg["TAU"])
    valid_criterion = nn.BCEWithLogitsLoss(weight=class_weight, reduction='none')
    optimizer = make_optimizer(cfg, model)

    if cfg["CUDA"]:
        model = model.cuda()
        criterion = criterion.cuda()
        kd_criterion.cuda()
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

    if cfg["MULTI_GPU"]:
        model = nn.DataParallel(model)

    # create data loaders & lr scheduler
    train_loader = DataLoader(train_ds, cfg["BATCH_SIZE"], 
                              pin_memory=False, shuffle=True,
                              drop_last=False, num_workers=cfg['NUM_WORKERS'])
    valid_loader = DataLoader(valid_ds, pin_memory=False,
                              shuffle=False, drop_last=False, 
                              num_workers=cfg['NUM_WORKERS'])
    test_loader = DataLoader(test_ds, pin_memory=False,
                             collate_fn=test_collate_fn, shuffle=False, 
                             drop_last=False, num_workers=cfg['NUM_WORKERS'])
    scheduler = WarmupCyclicalLR("cos", cfg["BASE_LR"], cfg["EPOCHS"],
                                 iters_per_epoch=len(train_loader),
                                 warmup_epochs=cfg["WARMUP_EPOCHS"])
    logger.info("Using {} lr scheduler\n".format(scheduler.mode))

    if args.eval:
        _, prob = validate(cfg, valid_loader, model, valid_criterion)
        imgids = pd.read_csv(cfg["DATA_DIR"] + "valid_{}_df_fold{}.csv" \
            .format(cfg["SPLIT"], cfg["FOLD"]))["image"]    
        save_df = pd.concat([imgids, pd.DataFrame(prob.numpy())], 1)
        save_df.columns = ["image", "any",
                           "intraparenchymal", "intraventricular",
                           "subarachnoid", "subdural", "epidural"]
        save_df.to_csv(os.path.join(cfg["OUTPUT_DIR"], "val_" + cfg["SESS_NAME"] + '.csv'), 
                       index=False)
        return
        
    if args.eval_test:
        if not os.path.exists(cfg["OUTPUT_DIR"]):
            os.makedirs(cfg["OUTPUT_DIR"])
        submit_fpath = os.path.join(cfg["OUTPUT_DIR"], "test_" + cfg["SESS_NAME"] + '.csv')
        submit_df = test(cfg, test_loader, model)
        submit_df.to_csv(submit_fpath, index=False)
        return

    for epoch in range(start_epoch, cfg["EPOCHS"]):
        logger.info("Epoch {}\n".format(str(epoch + 1)))
        random.seed(epoch)
        torch.manual_seed(epoch)
        # train for one epoch
        train(cfg, train_loader, 
              model, criterion, 
              kd_criterion, optimizer, 
              scheduler, epoch)
        # evaluate
        loss, _ = validate(cfg, valid_loader, model, valid_criterion)
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
    # merge configs
    cfg = configs
    args = parser.parse_args()
    opts = {}
    for full_key, v in zip(args.opts[0::2], args.opts[1::2]):
        opts[full_key] = ast.literal_eval(v)
    cfg.update(opts)

    # create logger
    global logger
    logger = setup_logger("3D Training", cfg["LOGS_DIR"], 
        cfg["LOCAL_RANK"], cfg["SESS_NAME"] + ".txt")
    logger.info("{}\n".format(args))
    logger.info("{}\n".format(cfg))
    
    main(cfg)
