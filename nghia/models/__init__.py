
import apex
from apex import amp
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import pandas as pd

from .octavte_lstm import OctResSLSTM
from .octave_resnet_crnn import OctResLSTM
from .octave_resnet import OctResNet50
from .octave_hybrid import OctResHybridLSTM
from .densenet import DenseNet
from .densenet_crnn import DenseNetLSTM
from .seresnext_crnn import SEResNeXT50LSTM
from .utils_module import *

def get_model(cfg):
    if cfg.TRAIN.MODEL == "octave-resnet50":
        model = OctResNet50
    elif cfg.TRAIN.MODEL == "octave-resnet50-lstm":
        model = OctResLSTM
    elif cfg.TRAIN.MODEL == "octave-resnet50-slstm":
        model = OctResSLSTM
    elif cfg.TRAIN.MODEL == "seresnext50-lstm":
        model = SEResNeXT50LSTM
    elif cfg.TRAIN.MODEL == "octave-resnet50-hybrid":
        model = OctResHybridLSTM
    elif cfg.TRAIN.MODEL.startswith("densenet"):
        if cfg.TRAIN.CRNN:
            model = DenseNetLSTM
        else:
            model = DenseNet
    else:
        return None
    
    if model:
        return model(model_name=cfg.TRAIN.MODEL, 
                     input_channels=cfg.DATA.INP_CHANNEL,
                     num_classes=cfg.TRAIN.NUM_CLASSES,
                     pretrained=False,
                     subtype_head=cfg.DATA.SUB_TYPE_HEAD,
                     cfg=cfg)
    else:
        print("Invalid model name")

def to_csv(cfg, filename, ids, probs):
    submit = pd.concat([pd.Series(ids), pd.DataFrame(probs)], axis=1)
    submit.columns = ["image", "any", "intraparenchymal", "intraventricular", "subarachnoid", "subdural", "epidural"]
    submit.to_csv(os.path.join(cfg.DIRS.OUTPUTS, filename), index=False)


def test_model(_print, cfg, model, test_loader, smooth=False, tta=False):
    model.eval()
    if smooth:
        _print("[Smooth Infer]")
    ids = []
    probs = []
    tbar = tqdm(test_loader)
    with torch.no_grad():
        for i, (image, id_code) in enumerate(tbar):
            
            if cfg.DATA.INP_CHANNEL == 3 and not cfg.DATA.ONE_SITE:
                image1, image2 = image
                image1 = image1.cuda()
                image2 = image2.cuda()
                output1 = model(image1)
                output2 = model(image2)
                output = (output1 + output2) / 2.

            else: 
                image = image.cuda()
                if tta:
                    output = hvflip_tta(model, image)
                else:
                    output = model(image)

            output = torch.sigmoid(output)
            if smooth:
                output = avgmvsmooth(output, gpu=True)
                output = watersmooth(output, gpu=True)
                
            probs.append(output.cpu().numpy())
            # ids += id_code
            ids += list(map(lambda x: x[0], id_code))
    
    probs = np.concatenate(probs, 0)
    to_csv(cfg, f"test_{cfg.EXP}.csv", ids, probs)

def valid_model(_print, cfg, model, valid_loader, valid_criterion, smooth_valid=False, tta=False):
    model.eval()
    # logits = []
    loss_array = []
    losses1 = []
    losses2 = []

    ids = []
    probs = []
    
    if smooth_valid:
        _print("[Smooth Valid]")
    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (image, target, id_code) in enumerate(tbar):
            target = target.cuda()
            if cfg.DATA.INP_CHANNEL == 3 and not cfg.DATA.ONE_SITE:
                image1, image2 = image
                image1 = image1.cuda()
                image2 = image2.cuda()
                output1 = model(image1)
                output2 = model(image2)
                output = (output1 + output2) / 2.
                losses1.append((valid_criterion(output1, target)).cpu().numpy())
                losses2.append((valid_criterion(output2, target)).cpu().numpy())
            else:
                if cfg.TRAIN.CRNN:
                    bs, ns, c, h, w = image.size()
                    # image = image.view(bs * ns, c, h, w)
                    target = target.view(bs * ns, -1)
                    
                image = image.cuda()
                
                if tta:
                    # output = tencrop_tta(model, image, crop_size=(cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE))
                    output = hvflip_tta(model, image)
                else:
                    output = model(image)
            
            output = torch.sigmoid(output)
            if smooth_valid:
                output = avgmvsmooth(output, gpu=True)
                output = watersmooth(output, gpu=True)
            probs.append(output.cpu().numpy())
            ids += list(map(lambda x: x[0], id_code))
            # Filter brain presence only
            # output = output[brain_id[0]]
            # target = target[brain_id[0]]
            
            loss = valid_criterion(output, target)
            loss_array.append(loss.cpu().numpy())

            # logits.append(output)
    
    probs = np.concatenate(probs, 0)
    # to_csv(cfg, f"valid_{cfg.EXP}.csv", ids, probs)

    loss_array = np.concatenate(loss_array, 0)
    if not cfg.DATA.ONE_SITE:
        losses1 = (np.concatenate(losses1, 0)).mean()
        losses2 = (np.concatenate(losses2, 0)).mean()
        _print(f"loss1: {losses1} - loss2: {losses2}")
    
    val_loss = loss_array.mean()
    
    if cfg.TRAIN.NUM_CLASSES  == 1:
        _print(f"Validation any loss: {val_loss:.5f}")
    else:
        any_loss = loss_array[:, 0].mean()
        intraparenchymal_loss = loss_array[:, 1].mean()
        intraventricular_loss = loss_array[:, 2].mean()
        subarachnoid_loss = loss_array[:, 3].mean()
        subdural_loss = loss_array[:, 4].mean()
        epidural_loss = loss_array[:, 5].mean()

        _print("Validation loss: {:.5f} - any: {:.3f} - intraparenchymal: {:.3f} - intraventricular: {:.3f} - subarachnoid: {:.3f} - subdural: {:.3f} - epidural: {:.3f}\n".format(
            val_loss, any_loss, intraparenchymal_loss, intraventricular_loss, subarachnoid_loss, subdural_loss, epidural_loss))
    
    return val_loss

def train_loop(_print, cfg, model, train_loader, criterion, valid_loader, valid_criterion, optimizer, scheduler, start_epoch, best_metric):
    
    for epoch in range(start_epoch, cfg.TRAIN.EPOCHS):
        _print(f"Epoch {epoch + 1}")

        losses = AverageMeter()
        model.train()
        tbar = tqdm(train_loader)

        for i, (image, target) in enumerate(tbar):
            image = image.cuda()
            target = target.cuda()

            if cfg.TRAIN.CRNN:
                bs, ns, c, h, w = image.size()
                # image = image.view(bs * ns, c, h, w)
                target = target.view(bs * ns, -1)

            # calculate loss
            if np.random.uniform() < cfg.DATA.CUTMIX_PROB:
                if cfg.TRAIN.CRNN:
                    image = image.view(bs * ns, c, h, w)
                mixed_x, y_a, y_b, lam = cutmix_data(image, target)
                if cfg.TRAIN.CRNN:
                    mixed_x = mixed_x.view(bs, ns, c, h, w)
                output = model(mixed_x)
                loss = mixup_criterion(criterion, output, y_a, y_b, lam)
            else:
                if cfg.TRAIN.MODEL == 'octave-resnet50-hybrid':
                    output, output_2d = model(image)
                    loss = (criterion(output, target) + criterion(output_2d, target)) / 2.
                else:
                    output = model(image)
                    loss = criterion(output, target)
            
            # gradient accumulation
            loss = loss / cfg.OPT.GD_STEPS
        
            with amp.scale_loss(loss, optimizer) as scaled_loss:
                scaled_loss.backward()
            
            if (i + 1) % cfg.OPT.GD_STEPS == 0:
                scheduler(optimizer, i, epoch, None) # Cosine LR Scheduler
                optimizer.step()
                optimizer.zero_grad()

            # record loss
            losses.update(loss.item() * cfg.OPT.GD_STEPS, image.size(0))
            tbar.set_description("Train loss: %.3f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))
        
        _print("Train loss: %.3f, learning rate: %.6f" % (losses.avg, optimizer.param_groups[-1]['lr']))

        val_loss = valid_model(_print, cfg, model, valid_loader, valid_criterion)
        is_best = val_loss < best_metric
        best_metric = min(val_loss, best_metric)
        
        # scheduler.step() # Cyclic scheduler

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': cfg.EXP,
            'state_dict': model.state_dict(),
            'best_metric': best_metric,
            'optimizer': optimizer.state_dict(),
        }, is_best, root=cfg.DIRS.WEIGHTS, filename=f"{cfg.EXP}.pth")