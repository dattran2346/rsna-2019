import torch
from torch import nn
import numpy as np

def reg_l2sp(model, model_source_weights):
    fea_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if 'head' not in name:
            fea_loss += 0.5 * torch.norm(param - model_source_weights[name]) ** 2
    return fea_loss

def reg_classifier(model):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if 'head' in name:
            l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def reg_l2sp_no_bias(model, model_source_weights):
    fea_loss = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if 'head' not in name:
            if 'bias' not in name:
                fea_loss += 0.5 * torch.norm(param.to(torch.float) - model_source_weights[name].to(torch.float)) ** 2
    return fea_loss

def reg_classifier_no_bias(model):
    l2_cls = torch.tensor(0.).cuda()
    for name, param in model.named_parameters():
        if 'head' in name:
            if 'bias' not in name:
                l2_cls += 0.5 * torch.norm(param) ** 2
    return l2_cls

def split_weights(net):
    """split network weights into to categlories,
    one are weights in conv layer and linear layer,
    others are other learnable paramters(conv bias, 
    bn weights, bn bias, linear bias)
    Args:
        net: network architecture
    
    Returns:
        a dictionary of params splite into to categlories
    """

    decay = []
    no_decay = []

    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            decay.append(m.weight)

            if m.bias is not None:
                no_decay.append(m.bias)
        
        else: 
            if hasattr(m, 'weight'):
                no_decay.append(m.weight)
            if hasattr(m, 'bias'):
                no_decay.append(m.bias)
        
    assert len(list(net.parameters())) == len(decay) + len(no_decay)

    return [dict(params=decay), dict(params=no_decay, weight_decay=0)]

def cutmix_data(inputs, targets, alpha=1.):
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
    lamb = 1 - ((x2 - x1) * (y2 - y1) / (inputs.size()[-1] * inputs.size()[-2]))
    return inputs, targets, targets[shuffled_idxs], lamb

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)