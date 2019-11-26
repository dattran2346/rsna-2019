from collections import OrderedDict
import logging
import math
import numpy as np
import os
import shutil
import sys
import torch
import torch.optim
import torch.distributed as dist


def test_collate_fn(batch):
    imgs, img_names = batch[0]
    imgs = imgs.unsqueeze(0)
    img_names = img_names.tolist()
    return imgs, img_names


def tta(model, image, seq_len=None):
    output = model(image, seq_len) + model(image.flip(2), seq_len) + model(image.flip(3), seq_len)
    one_third = float(1.0 / 3.0)
    return output * one_third


def accuracy(output, target, topk=(1,)):
    """
    Computes the accuracy over the k top predictions for the specified values of k.
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state, is_best, root, filename):
    """
    Saves checkpoint and best checkpoint (optionally)
    """
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(
            os.path.join(
                root, filename), os.path.join(
                root, 'best_' + filename))


def load_state_dict(state_dict, model):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    try:
        model.load_state_dict(new_state_dict)
    except:
        # load previous state dict after creating recurrent cell
        for key, value in model.state_dict().items():
            if key.startswith('decoder.'):
                new_state_dict[key] = value
        model.load_state_dict(new_state_dict)
        

def make_optimizer(cfg, model):
    """
    Create optimizer with per-layer learning rate and weight decay.
    """
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = cfg["BASE_LR"]
        weight_decay = cfg["WEIGHT_DECAY"]
        if "bias" in key:
            weight_decay = cfg["WEIGHT_DECAY_BIAS"]
            lr = cfg["BASE_LR"] * cfg["BIAS_LR_FACTOR"]
        params += [{"params": [value], "lr": lr, "weight_decay": weight_decay}]

    optimizer = torch.optim.AdamW(params, lr, eps=1e-3)
    return optimizer


def mixup_data(x, y, alpha=1.0, use_cuda=True):
    """
    Returns mixed inputs, pairs of targets, and lambda.
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def cutmix_data(inputs, targets, alpha=1.):
    """
    Returns cut-mixed inputs, pairs of targets, and lambda.
    """
    bsize, _, h, w = inputs.shape
    shuffled_idxs = torch.randperm(bsize).cuda()

    inputs_s = inputs[shuffled_idxs]
    lamb = np.random.beta(alpha, alpha)

    rx = np.random.randint(w)
    ry = np.random.randint(h)
    cut_ratio = np.sqrt(1. - lamb)
    rw = np.int(cut_ratio * w)
    rh = np.int(cut_ratio * h)

    x1 = np.clip(rx - rw // 2, 0, w)
    x2 = np.clip(rx + rw // 2, 0, w)
    y1 = np.clip(ry - rh // 2, 0, h)
    y2 = np.clip(ry + rh // 2, 0, h)

    inputs[:, :, x1:x2, y1:y2] = inputs_s[:, :, x1:x2, y1:y2]
    # adjust lambda to exactly match pixel ratio
    lamb = 1 - ((x2 - x1) * (y2 - y1) /
                (inputs.size()[-1] * inputs.size()[-2]))
    return inputs, targets, targets[shuffled_idxs], lamb


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    """
    Mixes loss from pairs of targets (y_a, y_b) based on lambda.
    """
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def synchronize():
    """
    Helper function to synchronize (barrier) among all processes when
    using distributed training.
    """
    if not dist.is_available():
        return
    if not dist.is_initialized():
        return
    world_size = dist.get_world_size()
    if world_size == 1:
        return
    dist.barrier()


def setup_logger(name, save_dir, distributed_rank, filename="log.txt"):
    """
    Source:
    
    https://github.com/facebookresearch/maskrcnn-benchmark/blob/master/maskrcnn_benchmark/utils/logger.py
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # don't log results for the non-master process
    if distributed_rank > 0:
        return logger
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger



class AverageMeter(object):
    """
    Computes and stores the average and current value.
    """
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
