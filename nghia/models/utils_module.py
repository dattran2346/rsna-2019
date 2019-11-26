import os
import shutil

import torch
import torch.nn as nn
import numpy as np 

def to_var(x, volatile=False):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=volatile)
    
def _initialize_weights(module):
    for m in module.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm1d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()


def _freeze_weights(module):
    for param in module.parameters():
        param.requires_grad = False



def save_checkpoint(state, is_best, root, filename):
    torch.save(state, os.path.join(root, filename))
    if is_best:
        shutil.copyfile(os.path.join(root, filename), os.path.join(root, 'best_' + filename))


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

def watersmooth(y_hat, gpu=False):
    output = y_hat.cpu().numpy()
    _dtype = output.dtype
    for d in range(output.shape[1]):
        res = output[:, d].copy()
        n = res.shape[0]
        for i in range(1, n-1, 1):
            # left_idx = np.argmax(res[:i])
            # right_idx = i + 1 + np.argmax(res[i+1:])
            left_idx, right_idx = i-1, i+1
            # if i > 1 and i < n-2:
            #     left_idx, right_idx = i-2, i+2
            to_fill = np.ones(right_idx-left_idx+1, dtype=_dtype) * min(res[left_idx], res[right_idx])
            res[left_idx:right_idx+1] = np.max([res[left_idx:right_idx+1], to_fill], 0)
        output[:, d] = res
    output = torch.from_numpy(output)
    return output.cuda() if gpu else output

def avgmvsmooth(y_hat, gpu=False):
    output = y_hat.cpu().numpy()
    _dtype = output.dtype
    for d in range(output.shape[1]):
        res = output[:, d].copy()
        n = res.shape[0]
        for i in range(1, n-1, 1):
            left_idx, right_idx = i-1, i+1
            res[i] = 1/6 * output[:,d][left_idx] + 2/3 * output[:,d][i] + 1/6 * output[:,d][right_idx] 
        output[:, d] = res
    output = torch.from_numpy(output)
    return output.cuda() if gpu else output   

def hvflip_tta(model, image):
    def torch_flip(x, dim):
        """
        Flip image tensor horizontally
        :param x:
        :return:
        """
        return x.flip(dim)

    output = (
        model(image)
        + model(torch_flip(image, -1))
        + model(torch_flip(image, -2))
    )

    one_over = float(1.0 / 3.0)
    return output * one_over

def inverse_hvflip_tta(model, image):
    def torch_flip(x, dim):
        """
        Flip image tensor horizontally
        :param x:
        :return:
        """
        return x.flip(dim)

    output = hvflip_tta(model, image)
    
    inv_image = image.flip(1)
    inv_output = (
        model(inv_image)
        + model(torch_flip(inv_image, -1))
        + model(torch_flip(inv_image, -2))
    )

    one_over = float(1.0 / 3.0)
    inv_output *= one_over

    return (output + inv_output.flip(0)) / 2.

def tencrop_tta(model, image, crop_size):
    """Test-time augmentation for image classification that takes five crops out of input tensor (4 on corners and central)
    and averages predictions from them and from their horisontally-flipped versions (10-Crop TTA).
    :param model: Classification model
    :param image: Input image tensor
    :param crop_size: Crop size. Must be smaller than image size
    :return: Averaged logits
    """
    image_height, image_width = int(image.size(-2)), int(image.size(-1))
    crop_height, crop_width = crop_size

    assert crop_height <= image_height
    assert crop_width <= image_width

    bottom_crop_start = image_height - crop_height
    right_crop_start = image_width - crop_width
    crop_tl = image[..., :crop_height, :crop_width]
    crop_tr = image[..., :crop_height, right_crop_start:]
    crop_bl = image[..., bottom_crop_start:, :crop_width]
    crop_br = image[..., bottom_crop_start:, right_crop_start:]

    assert crop_tl.size(-2) == crop_height
    assert crop_tr.size(-2) == crop_height
    assert crop_bl.size(-2) == crop_height
    assert crop_br.size(-2) == crop_height

    assert crop_tl.size(-1) == crop_width
    assert crop_tr.size(-1) == crop_width
    assert crop_bl.size(-1) == crop_width
    assert crop_br.size(-1) == crop_width

    center_crop_y = (image_height - crop_height) // 2
    center_crop_x = (image_width - crop_width) // 2

    crop_cc = image[
        ...,
        center_crop_y : center_crop_y + crop_height,
        center_crop_x : center_crop_x + crop_width,
    ]
    assert crop_cc.size(-2) == crop_height
    assert crop_cc.size(-1) == crop_width

    def torch_fliplr(x):
        """
        Flip image tensor horizontally
        :param x:
        :return:
        """
        return x.flip(-1)

    output = (
        model(crop_tl)
        + model(torch_fliplr(crop_tl))
        + model(crop_tr)
        + model(torch_fliplr(crop_tr))
        + model(crop_bl)
        + model(torch_fliplr(crop_bl))
        + model(crop_br)
        + model(torch_fliplr(crop_br))
        + model(crop_cc)
        + model(torch_fliplr(crop_cc))
    )

    one_over_10 = float(1.0 / 10.0)
    return output * one_over_10

class AverageMeter(object):
    """Computes and stores the average and current value"""

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