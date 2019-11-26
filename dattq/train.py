import argparse
import ast
import gc
from PIL import Image
import itertools
import math
import numpy as np
import os
import pandas as pd
import random
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
import time
from tqdm import tqdm
import warnings

from apex import amp

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy("file_system")

from lr_scheduler import LR_Scheduler
from model import ResNet
from efficientnet_pytorch import EfficientNet
from utils import AverageMeter, save_checkpoint, load_state_dict, set_random_seed

from datasets import get_train_dl, get_val_dl

import warnings
warnings.filterwarnings('ignore')

# CLASS_WEIGHT = torch.Tensor([0.285684121621622, 0.142857142857143, 0.142857142857143, 0.142857142857143, 0.142857142857143, 0.142857142857143])

#################################################
# Yet another temp stuff, cut mix
#################################################
def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
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
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


torch.autograd.set_detect_anomaly(True)
#################################################
# FC LSTM Decoder
from collections import OrderedDict
class Decoder(nn.Module):

    def __init__(self, args):
        super(Decoder, self).__init__()
        self.args = args
        self.lstm = nn.LSTM(input_size=args.nfeatures, hidden_size=args.nfeatures//4, num_layers=2, batch_first=True, dropout=0, bidirectional=True) # drop not good
        self.fc = nn.Linear(args.nfeatures//2, args.nclasses)

        ## these init is also not good :))
        # # ### Zero init
        # # bias_ih_l[k] (b_ii|b_if|b_ig|b_io)`
        # # bias_hh_l[k] (b_hi|b_hf|b_hg|b_ho)
        # self.lstm.bias_ih_l0.data.zero_()
        # self.lstm.bias_hh_l0.data.zero_()
        # self.lstm.bias_ih_l1.data.zero_()
        # self.lstm.bias_hh_l1.data.zero_()
        # self.fc.bias.data.zero_()

        # ### Set forget bias to 1, suggest by `An Empirical Exploration of Recurrent Network Architectures`
        # self.lstm.bias_ih_l0.data[512*1:512*2].fill_(1.)
        # self.lstm.bias_hh_l0.data[512*1:512*2].fill_(1.)
        # self.lstm.bias_ih_l1.data[512*1:512*2].fill_(1.)
        # self.lstm.bias_hh_l1.data[512*1:512*2].fill_(1.)

        # ### xavier init, suggest by https://danijar.com/language-modeling-with-layer-norm-and-gru/
        # ### default init is uniform distribution
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l0.data)
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l0.data)
        # nn.init.xavier_uniform_(self.lstm.weight_ih_l1.data)
        # nn.init.xavier_uniform_(self.lstm.weight_hh_l1.data)

    def forward(self, x):
        # input: (args.bs*args.nslices, args.nfeatures)
        # output: (args.bs*args.nslices, args.nclasses)
        # view dont support back prop, reshape ok ??
        x, _ = self.lstm(x) # x (bs, nslices, nfeatures)
        x = self.fc(x)
        return  x

#################################################
# Convolutional LSTM
from convlstm import ConvLSTM

class ConvDecoder(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.lstm = ConvLSTM(input_size=(args.image_size//32, args.image_size//32), 
                        kernel_size=(3, 3), input_dim=args.nfeatures, hidden_dim=args.nfeatures//4, num_layers=2, 
                        batch_first=True, return_all_layers=False)
        self.fc = nn.Linear(args.nfeatures//4, args.nclasses)
    
    def forward(self, x):
        # x: (bs, nslices, nfeatures, h, w)
        layer_outputs, _ = self.lstm(x)
        x = layer_outputs[0] # -> (bs, nslices, nfeatues//4, h, w)
        x = x.mean(dim=(3, 4)) # -> adaptive average pool in spatital dimension
        return self.fc(x) # -> bs, nslices, nclasses

#################################################
# Temp stuff
#################################################

def make_optimizer(args, model):
    lr = args.lr
    # weight_decay = 1e-2
    # weight_decay_bias = 0

    params_list = []
    if args.input_level == 'per-study':
        encoder, decoder = model
        # for m in model:
        #     for key, value in m.named_parameters():
        #         print(key)
        #         if not value.requires_grad:
        #             continue
        #         if 'bias' in key:
        #             # x2 lr for bias, and weight decay for bias is 0
        #             params_list.append({'params': [value], 'lr': lr*2, 'weight_decay': weight_decay_bias})
        #         else:
        #             # weight decay for weight is 1e-2
        #             params_list.append({'params': [value], 'lr': lr, 'weight_decay': weight_decay})

        params_list.append({'params': encoder.parameters()})
        params_list.append({'params': decoder.parameters()})
    else:
        params_list.append({'params': model.parameters()})

    if args.optim == 'adam':
        optimizer = torch.optim.Adam(params_list, lr=lr, eps=1e-3)
    elif args.optim == 'radam':
        from radam import RAdam
        optimizer = RAdam(params_list, lr=lr, eps=1e-3)
    elif args.optim == 'adamw':
        from radam import AdamW
        optimizer = AdamW(params_list, lr=lr, eps=1e-3)
    elif args.optim == 'sgd':
        # lr = lr*100 a dat suggest, sgd is much slower than adam
        optimizer = torch.optim.SGD(params_list, lr=lr, momentum=0.9)
    else:
        raise ValueError("Unkniown optimizer")

    if args.lookahead:
        from lookahead import Lookahead
        optimizer = Lookahead(optimizer)

    return optimizer


########################################################
# Remove nan hook
########################################################
# class RemoveNanGradHook():
#     def __init__(self, module):
#         self.hook = module.register_backward_hook(self.hook_fn)

#     def hook_fn(self, module, grad_in, grad_out):
#         print(f'#input: {len(grad_in)}, #output: {len(grad_out)}')
#         nan_index = torch.isnan(grad_out[0])
#         if nan_index.sum() > 0:
#             print(f'Nan occur at {module}, set nan to 0')
#             grad_out[nan_index] = 0
#             return grad_out

#     def close(self):
#         self.hook.remove()


def main(args):
    best_loss = float('inf')
    global logger

    
    best_epoch = 0

    set_random_seed(args.seed)

    # create model
    if "resnet" in args.backbone or "resnext" in args.backbone:
        model = ResNet(args)
    # elif 'b' in args.backbone:
    #     model = EfficientNet.from_pretrained(f'efficientnet-{args.backbone}', 8)
    # elif 'd' in args.backbone:
    #     from densenet import DenseNet
    #     model = DenseNet(args)

    if args.input_level == 'per-study':
        # add decoder if train per-study

        if args.conv_lstm:
            decoder = ConvDecoder(args)
        else:
            decoder = Decoder(args)
        
        encoder = model
        model = (encoder, decoder)

        # decoder_hooks = [RemoveNanGradHook(m) for name, m in decoder._modules.items()]
        # encoder_hookds = [RemoveNanGradHook(m) for name, m in encoder._modules.items()]


    criterion = nn.BCEWithLogitsLoss(weight=args.class_weight, reduction='none')

    optimizer = make_optimizer(args, model)

    if args.input_level == 'per-study':
        model[0].cuda(), model[1].cuda()
    else:
        model = model.cuda()
    criterion = criterion.cuda()
    model, optimizer = amp.initialize(list(model),
                                        optimizer,
                                        opt_level=args.opt_level,
                                        verbosity=0, # do not print that shit out
                                        keep_batchnorm_fp32=True)

    train_loader = get_train_dl(args)
    val_loader = get_val_dl(args)

    scheduler = LR_Scheduler('cos', base_lr=args.lr, num_epochs=args.epochs, iters_per_epoch=len(train_loader), warmup_epochs=args.warmup)

    ####################################
    # finetune from checkpoint
    ####################################
    if args.finetune:
        if os.path.isfile(args.finetune):
            print("=> loading checkpoint '{}'".format(args.finetune))
            checkpoint = torch.load(args.finetune, "cpu")
            # load model
            input_level = checkpoint['input_level']
            assert input_level == args.input_level

            if args.input_level == 'per-study':
                encoder, decoder = model
                load_state_dict(checkpoint.pop('encoder'), encoder)
                load_state_dict(checkpoint.pop('decoder'), decoder)
            else:
                load_state_dict(checkpoint.pop('state_dict'), model)
            print("=> Finetune checkpoint '{}'".format(args.finetune))


    ####################################
    # resume from a checkpoint
    ####################################
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, "cpu")
            input_level = checkpoint['input_level']
            assert input_level == args.input_level

            if args.input_level == 'per-study':
                encoder, decoder = model
                load_state_dict(checkpoint.pop('encoder'), encoder)
                load_state_dict(checkpoint.pop('decoder'), decoder)
            else:
                load_state_dict(checkpoint.pop('state_dict'), model)

            optimizer.load_state_dict(checkpoint.pop('optimizer'))
            args.start_epoch = checkpoint['epoch']+1 # start from prev + 1
            best_epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    # no logging when evaluate only
    if args.evaluate:
        val_loss, val_losses = validate(val_loader, model)
        print(f"Evaluation loss: {val_loss}\t")
        return

    ##############################3
    # Only log when training
    from logger import TrainingLogger
    logger = TrainingLogger(args)


    for epoch in range(args.start_epoch, args.epochs):
        # print('EPOCH:', epoch)
        logger.on_epoch_start(epoch)
        ####################################
        # train for one epoch
        ####################################
        train_losses, lr = train(train_loader, model, criterion, optimizer, scheduler, epoch)

        ####################################
        # evaluate
        ####################################
        val_loss, val_losses = validate(val_loader, model)
        loss = val_loss

        # remember best accuracy and save checkpoint
        is_best = loss < best_loss
        best_loss = min(loss, best_loss)

        # save checkpoint to resume training
        checkpoint = {
            'epoch': epoch, # next epoch
            'best_loss': best_loss,
            'optimizer': optimizer.state_dict(),
            'input_level': args.input_level
        }

        if args.input_level == 'per-study':
            encoder, decoder = model
            checkpoint['encoder'] = encoder.state_dict()
            checkpoint['decoder'] = decoder.state_dict()
        else:
            checkpoint['state_dict'] = model.state_dict()

        save_checkpoint(checkpoint, is_best, checkname=args.checkname, epoch=epoch, save_all=args.save_all)

        # save which epoch is best
        if is_best:
            best_epoch = epoch
        
        logger.on_epoch_end(epoch, lr, train_losses, val_losses)

        # something leak here??
        del train_losses, val_loss, val_losses
        import gc
        gc.collect()

    logger.on_training_end(best_loss, best_epoch)
    print(f'====== Finish training, best loss {best_loss:.5f}@e{best_epoch+1} ======')


def train(train_loader, model, criterion, optimizer, scheduler, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    if args.input_level == 'per-study':
        encoder, decoder = model
        encoder.train()
        decoder.train()
    else:
        model.train()

    tbar = tqdm(train_loader)
    for i, (images, targets) in enumerate(tbar):
        # if i == 10: break
        images = images.cuda()
        targets = targets.cuda()

        ########################################
        # Train
        ########################################
        # per-study: images (bs, nslices, nwindows, h, w), targets (bs, nslices, nclasses)
        # if train per-study, stack bs and nslices=10 -> new bs
        images = images.view(args.batch_size*args.nslices, args.mix_window, args.image_size, args.image_size)
        targets = targets.view(args.batch_size*args.nslices, args.nclasses)

        # if not args.conv_lstm:
        #     if np.random.uniform() < 0.5:
        #         x, y_a, y_b, lam = mixup_data(images, targets)
        #         x = encoder(x)
        #         x = x.reshape(args.batch_size, args.nslices, args.nfeatures)
        #         x = decoder(x)
        #         outputs = x.reshape(args.batch_size*args.nslices, args.nclasses)
        #         individual_loss = mixup_criterion(criterion, outputs, y_a, y_b, lam).mean(0)
        #         loss = individual_loss.mean()
        #     else:
        #         x = encoder(images)
        #         x = x.reshape(args.batch_size, args.nslices, args.nfeatures)
        #         x = decoder(x)
        #         outputs = x.reshape(args.batch_size*args.nslices, args.nclasses)
        #         individual_loss = criterion(outputs, targets).mean(0)
        #         loss = individual_loss.mean()
        # else:
        if np.random.uniform() < 0.5:
            x, y_a, y_b, lam = mixup_data(images, targets)
            x = encoder(x)
            # print('Encoder', x.shape)
            if args.conv_lstm:
                x = x.reshape(args.batch_size, args.nslices, args.nfeatures, args.image_size//32, args.image_size//32)
            else:
                x = x.reshape(args.batch_size, args.nslices, args.nfeatures)
            x = decoder(x)
            outputs = x.reshape(args.batch_size*args.nslices, args.nclasses)
            individual_loss = mixup_criterion(criterion, outputs, y_a, y_b, lam).mean(0)
            loss = individual_loss.mean()
        else:
            x = encoder(images)
            # print('Encoder', x.shape)
            if args.conv_lstm:
                x = x.reshape(args.batch_size, args.nslices, args.nfeatures, args.image_size//32, args.image_size//32)
            else:
                x = x.reshape(args.batch_size, args.nslices, args.nfeatures)
            x = decoder(x)
            outputs = x.reshape(args.batch_size*args.nslices, args.nclasses)
            individual_loss = criterion(outputs, targets).mean(0)
            loss = individual_loss.mean()


        ########################################
        # Accumulate gradient
        ########################################
        loss = loss / args.gds

        with amp.scale_loss(loss, optimizer) as scaled_loss:
            try:
                scaled_loss.backward() # test backward: nan torch.tensor(float('nan')).cuda()
            except Exception as e:
                print(e)
                optimizer.zero_grad() ## <- clear all corrupted accumulate gradient,
                ## this is essential, but for large model, there can be too many nan -> can't train at all, for r50: about 5 nan per epoch. late epoch occur much more often
                ## loss is normal, but CudnnBackward get nan

        if (i + 1) % args.gds == 0:
            scheduler(optimizer, i, epoch, None) # scheduler.step() before optimizer.step()
            ## Test 2: clip norm before step

            if args.input_level == 'per-study':
                torch.nn.utils.clip_grad_norm_(model[0].parameters(), args.clipnorm)
                torch.nn.utils.clip_grad_norm_(model[1].parameters(), args.clipnorm)
            else:
                torch.nn.utils.clip_grad_norm_(model[0].parameters(), args.clipnorm)

            # update model weight
            optimizer.step()
            optimizer.zero_grad()

        ########################################
        # record loss and show log
        ########################################'
        with torch.no_grad():
            losses.update(individual_loss.detach().cpu().numpy() * args.gds, images.size(0))
            loss_desc = ','.join([f'{n}: {l:.3f}' for n, l in zip(args.class_name, losses.avg)])

            tbar.set_description('%s, learning rate: %.4f'
                % (loss_desc,
                optimizer.param_groups[0]['lr']))

    # print(f"Training loss: {losses.avg.mean()}\t")
    return losses.avg, optimizer.param_groups[0]['lr']


def validate(valid_loader, model):
    # switch to evaluate mode
    import torch.nn.functional as F
    losses = AverageMeter()

    if args.input_level == 'per-study':
        encoder, decoder = model
        encoder.eval()
        decoder.eval()
    else:
        model.eval()

    y_pred = []
    y_true = []

    tbar = tqdm(valid_loader)
    with torch.no_grad():
        for i, (images, targets) in enumerate(tbar):
            # if i == 10: break
            images = images.cuda()

            # images: (1, tta, nslces, c, h, w)
            # targets: (1, nslices, nclasses)
            images.squeeze_(0) # (tta, nslces, c, h, w)
            targets.squeeze_(0) # (nslices, nclasses)

            tta, nslices, c, h, w = images.size()
            images = images.reshape(tta*nslices, c, h, w)
            
            x = encoder(images) # (tta*nslices, nfeatures)
            
            # fc lstm
            # x = x.reshape(tta, nslices, args.nfeatures) # -> (tta, nslices, nfeautures)
            
            if args.conv_lstm:
                x = x.reshape(tta, nslices, args.nfeatures, args.image_size//32, args.image_size//32)
            else:
                x = x.reshape(tta, nslices, args.nfeatures)

            outputs = decoder(x) # (tta, nslices, nclasses)
            outputs = outputs.mean(0) # avg logit of tta -> (nslices, nclasses)

            individual_loss = F.binary_cross_entropy_with_logits(outputs.cpu(), targets, args.class_weight, reduction='none').mean(0) # mean by batch size

            # val per slices, not per study
            losses.update(individual_loss.detach().cpu().numpy(), images.size(0))

    # print("Validation results:\t")
    # print(f"Loss: {losses.avg.mean():.5f}")

    return losses.avg.mean(), losses.avg


if __name__ == '__main__':
    from options import Options
    opt = Options()
    args = opt.parse()
    main(args)



