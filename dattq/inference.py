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
from sklearn.metrics import accuracy_score
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

from losses import SigmoidFocalLoss
from lr_scheduler import LR_Scheduler
from model import ResNet
from utils import AverageMeter, save_checkpoint, load_state_dict, set_random_seed

from datasets import get_test_dl, get_val_dl
from pathlib import Path
from efficientnet_pytorch import  EfficientNet
from train import Decoder, ConvDecoder


def main(args):
    # create model
    if "resnet" in args.backbone or "resnext" in args.backbone:
        print('resnet', args.att)
        model = ResNet(args)
    elif 'b' in args.backbone:
        model = EfficientNet.from_pretrained(f'efficientnet-{args.backbone}', 8)

    if args.input_level == 'per-study':
        # add decoder if train per-study
        if args.conv_lstm:
            decoder = ConvDecoder(args)
        else:
            decoder = Decoder(args)

        encoder = model
        model = (encoder, decoder)

    if args.input_level == 'per-study':
        model[0].cuda(), model[1].cuda()
    else:
        model = model.cuda()

    # optionally resume from a checkpoint
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

            # load_state_dict(checkpoint.pop('state_dict'), model)
            epoch = checkpoint['epoch']
            best_loss = checkpoint['best_loss']
            print(f"=> loaded checkpoint '{args.resume}' (loss {best_loss:.4f}@{epoch})")
        else:
            raise ValueError("=> no checkpoint found at '{}'".format(args.resume))

    # if args.to_stack:
    #     loader = get_test_dl(args)
    #     to_submit(args, model, loader)
    # else:
    if args.val:
        val_dl = get_val_dl(args)
        to_stacking_on_val(args, model, val_dl)
    else:
        test_dl = get_test_dl(args)
        to_stacking_on_test(args, model, test_dl)

def to_submit(args, model, test_loader):
    # switch to evaluate mode
    if args.input_level == 'per-study':
        encoder, decoder = model
        encoder.eval()
        decoder.eval()
    else:
        model.eval()

    inference_file = Path('inference')/f'{args.checkname}.csv'
    print(f'Save inference to file {inference_file}')

    with open(inference_file, 'w') as f:
        f.write('ID,Label\n')

        # make prediction on image w/ brain
        tbar = tqdm(test_loader)
        with torch.no_grad():
            for i, (images, ids) in enumerate(tbar):
                images = images[0]
                ids = ids[0]
                images = images.cuda()

                if args.input_level == 'per-study':
                    # infer 1 study, one at the time, treat nslices as bs
                    images.squeeze_(0) # (tta, nslices, window, h, w)
                    tta, nslices, c, h, w = images.size()
                    images = images.reshape(tta*nslices, c, h, w)

                    x = encoder(images) # (tta*nslices, nfeatures)

                    if args.conv_lstm:
                        x = x.reshape(tta, nslices, args.nfeatures, args.nfeatures//32, args.nfeatures//32)
                    else:
                        x = x.reshape(tta, nslices, args.nfeatures)

                    # x.unsqueeze_(0) # -> (1, nslices, nfeatures)
                    outputs = decoder(x) # -> (tta, nslices, nclasses)
                    # outputs.squeeze_(0) # (nslices, nclass)
                    outputs = outputs.mean(0) # -> (nslices, nclasses)
                    # Write logits

                    if args.infer_sigmoid:
                        outputs = torch.sigmoid(outputs)
                else:
                    outputs = model(images)
                    if args.infer_sigmoid:
                        outputs = torch.sigmoid(model(images))

                outputs = outputs.cpu().numpy()
                for _id, output in zip(ids, outputs):
                    for disease, prob in zip(args.class_name, output):
                        f.write(f'{_id}_{disease},{prob}\n')

        if args.infer_sigmoid:
            # image with no brain -> predict all 0
            # infer sigmoid, infer 0 directly
            df = pd.read_csv(Path(args.data_dir)/'fold/testset_stage1.csv')
            df = df[~df.BrainPresence]
            for _id in df.image.values:
                for disease in args.class_name:
                    f.write(f'{_id}_{disease},0\n')


def to_stacking_on_val(args, model, val_dl):
    # assert args.tta == 3 and args.input_level == 'per-study'
    val_file = Path('stacking')/f'{args.checkname}.csv'

    print(f'Save mode {val_file} for stacking')
    # make prediction on image w/ brain
    prediction = run_stacking_inference(args, model, val_dl)

    # make prediction on image w/o brain
    fold_path = Path(args.data_dir)/'fold'
    df = pd.read_csv(fold_path/"trainset_stage1_split_patients.csv")
    df = df[~df.BrainPresence]
    no_brain_val_df =  df.loc[df['fold'] == args.fold]
    
    prediction['image'] += list(no_brain_val_df.image.values)
    for disease in args.class_name:
        prediction[disease] += [0]*len(no_brain_val_df.image)
    
    df = pd.DataFrame(prediction)
    df.to_csv(val_file, index=None)
    

def to_stacking_on_test(args, model, test_dl):
    # assert args.tta == 3 and args.input_level == 'per-study'
    test_file = Path('stacking')/f'{args.checkname}.csv'
    print(f'Save mode {test_file} for stacking')

    # make prediction on image w/ brain 
    prediction = run_stacking_inference(args, model, test_dl)

    # make prediction on image w/o brain
    df = pd.read_csv(Path(args.data_dir)/'fold/testset_stage2.csv')
    no_brain_test_df = df[~df.BrainPresence]

    prediction['image'] += list(no_brain_test_df.image.values)
    for disase in args.class_name:
        prediction[disase] += [0]*len(no_brain_test_df.image)

    
    df = pd.DataFrame(prediction)
    df.to_csv(test_file, index=None)

def run_stacking_inference(args, model, dl):
    # switch to evaluate mode
    encoder, decoder = model
    encoder.eval()
    decoder.eval()

    from collections import defaultdict
    predictions = defaultdict(lambda: [])

    tbar = tqdm(dl)
    with torch.no_grad():
        for i, (images, ids) in enumerate(tbar):
            images = images[0]
            ids = ids[0]
            images = images.cuda()

            # infer 1 study, one at the time, treat nslices as bs
            images.squeeze_(0) # (tta, nslices, window, h, w)
            tta, nslices, c, h, w = images.size()
            images = images.reshape(tta*nslices, c, h, w)

            x = encoder(images) # (tta*nslices, nfeatures)

            if args.conv_lstm:
                x = x.reshape(tta, nslices, args.nfeatures, args.nfeatures//32, args.nfeatures//32)
            else:
                x = x.reshape(tta, nslices, args.nfeatures)

        # x.unsqueeze_(0) # -> (1, nslices, nfeatures)
            outputs = decoder(x) # -> (tta, nslices, nclasses)
            # outputs.squeeze_(0) # (nslices, nclass)
            outputs = outputs.mean(0) # -> (nslices, nclasses)
            # Write logits

            outputs = torch.sigmoid(outputs)

            outputs = outputs.cpu().numpy() # (nslices, nclasses)

            predictions['image'] += list(ids)

            for i in range(len(args.class_name)):
                disease = args.class_name[i]
                prob = outputs[:, i]
                predictions[disease] += list(prob)
        
        return predictions


if __name__ == '__main__':
    from options import Options
    options = Options()
    # options.parser.add_argument('--to-stack', default=False, action='store_true', help='To stack format for a dungnb')
    options.parser.add_argument('--val', default=False, action='store_true', help='Run on val dataset and output csv')
    args = options.parse()
    args.inference = True # run this on inference mode -> val dl return image names instead of labels
    main(args)
