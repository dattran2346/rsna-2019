import argparse
import os

class Options():
    def __init__(self):
        parser = argparse.ArgumentParser(
            description='PyTorch Training')

        #########################################################
        # Use as default
        #########################################################
        parser.add_argument('--seed', default=25, type=int,
                            help='Random seed for numpy and pytorch')
        parser.add_argument('--dtype', default='float16', type=str,
                            help='full/mixed precision training')
        parser.add_argument('--opt_level', default="O2", type=str)
        parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,
                            help="Modify config options using the command-line")
        parser.add_argument("--workers", default=8, type=int,
                            help="Number of workers")

        #########################################################
        # Data options
        #########################################################
        parser.add_argument("--data-dir", default=None, type=str,
                            help="Root data dir")
        # parser.add_argument("--data-variant", default='stage_1_train_images_C40_L80', type=str,
        #                     help="data variant to train")
        parser.add_argument("--image-size", default=256, type=int,
                            help='Input image size')
        parser.add_argument("--data-frac", default=1., type=float,
                            help="Use fraction of the data")

        #########################################################
        # Training option
        #########################################################
        parser.add_argument("--fold", default=0, type=int,
                            help="Fold")
        parser.add_argument("--epochs", default=30, type=int,
                            help="Number of epoch")
        parser.add_argument("--stop-epoch", default=None, type=int,
                            help='Stop training at epoch')
        parser.add_argument("--lr", default=1e-3, type=float,
                            help="base learning rate for bs=16")

        # batch
        parser.add_argument("--batch-size", default=16, type=int,
                            help="Batch size")
        parser.add_argument("--gds", default=1, type=int,
                            help="Use gradient accumulation")

        # save stop restart
        parser.add_argument("--checkname", default=None, type=str,
                            help="Save model name")
        parser.add_argument('--finetune', default=None, type=str,
                            help='Fine tune model on larger image size')
        parser.add_argument("--resume", default=None, type=str,
                            help="resume training from chkpt")
        parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                            help='evaluate model on validation set')
        parser.add_argument('--cutmix-prob', default=1., type=float,
                            help='cutmix probability')
        parser.add_argument("--infer", default=False, action='store_true',
                            help='Run inference after training')

        # optimizer
        parser.add_argument('--optim', default='adam', type=str,
                            help='Set optimizer')
        parser.add_argument('--lookahead', default=False, action='store_true',
                            help='Add look ahead wrapper')
        # scheduler
        parser.add_argument('--sched', default='cos', type=str,
                            help='Set custom scheduler')
        parser.add_argument('--warmup', default=None, type=int,
                            help='warmup epochs')

        # clip norm to avoid exploding gradient
        parser.add_argument('--clipnorm', default=1, type=int,
                            help='clip grad norm')

        #########################################################
        # Model specific options
        #########################################################
        parser.add_argument('--att', default="noop", type=str,
                            help='Add attention module (cbam, scse, ssa, gc)')
        parser.add_argument("--backbone", default='resnet50',
                            help="backbone to use")


        #########################################################
        # Project specific options, fix at the start of project
        #########################################################
        parser.add_argument('--nclasses', default=6, type=int,
                            help='Add look ahead wrapper')

        # extra-axial haemorrhage, intra-axial haemorrhage
        parser.add_argument('--class-name', default=['any', 'epidural', 'subdural', 'subarachnoid', 'intraparenchymal', 'intraventricular'],
                            help='Class name column in csv file')
        parser.add_argument('--class-weight', default=[2, 1, 1, 1, 1, 1],
                            help='Weight for each class')
        # parser.add_argument('--sampler', default='under', type=str,
        #                     help='Sampling base on under (epidural) or sampler (subdural), balance (total neg), all (all dataest)')

        # use mix window
        parser.add_argument('--mix-window', default=3, type=int,
                            help='Mix different window level and width, 1: use single brain window, 3: brain, subdural, bony, 6: all channel')

        # remove easy image and no brain image
        # parser.add_argument('--no-brain', default=False, action='store_true', ## Move to default
        #                     help='Remove no brain image from training')
        parser.add_argument('--cherry_pick', default=False, action='store_true',
                            help='Remove easy image from training')
        parser.add_argument('--cherry_df', default="hard_train.csv",
                            help='Cherry pick sample from datnt')
        parser.add_argument('--remove-to', default=None, type=int,
                            help='Gradually remove no brain and easy image to epochs')


        #########################################################
        # train per studies using RNN decoder                   #
        #########################################################
        parser.add_argument('--input-level', default='per-study',
                            help='Input per-slice or per-study')
        parser.add_argument('--nslices', default=10,
                            help='If train per-study, # slices for each study')
        parser.add_argument('--lstm-dropout', default=0,
                            help='Dropout for lstm')
        parser.add_argument('--conv-lstm', default=False, action='store_true',
                            help='Use convolutional LSTM')
        parser.add_argument('--cut-block', default=False, action='store_true',
                            help='Cut last block in layer 4 resnet')
        parser.add_argument('--drop-block', default=False, action='store_true',
                            help='Apply drop block to c3, c4')
        parser.add_argument('--aux', default=False, action='store_true',
                            help='Add auxilary layer for c3, c4')


        #########################################################
        # inference options
        #########################################################
        parser.add_argument('--infer-sigmoid', default=False, action='store_true',
                            help='Run inference output sigmoid (true) or logit (false)')
        parser.add_argument('--tta', default=1, type=int, 
                            help='Run w/ tta, default=1 -> no tta')
        parser.add_argument('--inference', default=False, action='store_true',
                            help='Run inference mode')

        #########################################################
        # Keep this shit ??
        #########################################################
        parser.add_argument('--start-epoch', default=0, type=int,
                            metavar='N', help='start epoch')
        parser.add_argument('--save-all', action='store_true', default=False,
                            help='Save all epoch')
        self.parser = parser

    def parse(self):
        args = self.parser.parse_args()

        # linear scaling learning rate
        args.lr = (args.lr / 16) * args.batch_size*args.gds

        # set warmup = 10% total epochs
        if args.warmup == None:
            args.warmup = int(args.epochs*0.1)

        # set remove default to x2 epochs after warmup
        if args.remove_to == None:
            args.remove_at = 2*args.warmup

        # set nfeatures
        if args.backbone in ['resnet18', 'resnet34']:
            args.nfeatures = 512
        elif args.backbone in ['resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d']:
            args.nfeatures = 2048

        # convert weight to tensor
        import torch
        args.class_weight = torch.Tensor(args.class_weight)
        return args

