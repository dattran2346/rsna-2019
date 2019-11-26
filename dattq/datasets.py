import numpy as np
import pandas as pd
import os
from PIL import Image

import torch
import torch.utils.data as D
from torch import Tensor

from torchvision import transforms
from pathlib import Path
import cv2
from PIL import Image
from augmentation import HarderAugmenter, ValAugmenter, TestAugmenter


STAGE_1_TRAIN_DIR = 'stage_1_train_images_L%d_W%d'
STAGE_1_TEST_DIR = 'stage_1_test_images_L%d_W%d'
STAGE_2_TEST_DIR = 'stage_2_test_images_L%d_W%d'


def get_init_conv_params_sigmoid(ww, wl, smooth=1., upbound_value=255.):
    """
    Source: https://github.com/MGH-LMIC/windows_optimization/blob/master/functions.py
    """
    w = 2./ww * math.log(upbound_value/smooth - 1.)
    b = -2.*wl/ww * math.log(upbound_value/smooth - 1.)
    return (w, b)

################################################
# May try if have time (stage 2) :))
################################################
# class WSO(nn.Module):
#     """
#     Window settings optimization.
#     Reference:
#         https://arxiv.org/pdf/1812.00572.pdf
#     """
#     def __init__(self, windows=OrderedDict({
#             'brain': {'W': 80, 'L': 40},
#             'subdural': {'W': 215, 'L': 75},
#             'bony': {'W': 2800, 'L': 600},
#             'tissue': {'W': 375, 'L': 40},
#             }), U=255., eps=1.):
#         super(WSO, self).__init__()
#         self.windows = windows
#         self.U = U
#         self.eps = eps
#         self.conv1x1 = nn.Conv2d(1, len(windows), kernel_size=1, stride=1, padding=0)
#         nn.init.ones_(self.conv1x1.weight.data)
#         nn.init.zeros_(self.conv1x1.bias.data)
#         weight, bias = self._get_window_params()
#         self.register_buffer("weight", weight)
#         self.register_buffer("bias", bias)

#     def _get_window_params(self):
#         weight = []
#         bias = []
#         for _, window in self.windows.items():
#             ww, wl = window["W"], window["L"]
#             w, b = get_init_conv_params_sigmoid(ww, wl, self.eps, self.U)
#             weight.append(w)
#             bias.append(b)
#         weight = torch.as_tensor(weight)
#         bias = torch.as_tensor(bias)
#         # print(weight, bias)
#         return weight, bias

#     def forward(self, x):
#         x = self.conv1x1(x)
#         if x.dtype == torch.float16:
#             self.weight = self.weight.half()
#             self.bias = self.bias.half()
#         x = x.mul(self.weight[None, :, None, None]) + self.bias[None, :, None, None]
#         x = torch.sigmoid(x)
#         x = x.mul(self.U)
#         return x

################################################
# Slices sampling strategy
################################################
# (40, 10) -> [ 0,  4, 14, 17, 18, 22, 25, 30, 32, 33]
def random_range_uniform_sorted(l, nslices):
    return np.sort(np.random.choice(l-1, nslices, replace=False),)


# (40, 10) -> [11, 34,  4, 32, 18, 27, 35, 20, 22, 17]
def random_range_uniform_unsorted(l, nslices):
    return np.random.choice(l-1, nslices, replace=False)


# (40, 10) -> [17, 18, 19, 20, 21, 22, 23, 24, 25, 26]
def contiguous_slice_sampling(l, nslices):
    if l > nslices:
        mid = np.random.randint(0, l)
        start = mid - nslices//2
        end = mid + (nslices//2 + nslices%2)

        # generate range and fix
        idx = np.arange(start, end)
        idx += (1 - min(0, idx[0])) # 0 -> 1: decrease performance, although not totally correct??
        idx -= ((max(l, idx[-1]) - l)+1)
        # idx[idx==-1] = (l-1) # just a wrapper
    else:
        idx = np.arange(l)
        pad = (nslices-l)//2
        idx = np.pad(idx, (pad, pad+l%2), mode='edge')
    return idx


class ImagesDS(D.Dataset):
    def __init__(self, args, aug, mode='train'):
        self.args = args
        self.mode = mode
        self.class_name = args.class_name
        self.aug = aug
        self.mix_window = self.args.mix_window

        if self.mix_window == 1:
            self.ct_windows = {'brain': {'L': 40, 'W': 80}}
        elif self.mix_window == 3:
            self.ct_windows = {
                    'brain': {'L': 40, 'W': 80}, # ok
                    'subdural': {'L': 75, 'W': 215},
                    'bony': {'L': 600, 'W': 2800}, # ok
                }
        elif self.mix_window == 6:
            self.ct_windows = {
                    'brain': {'L': 40, 'W': 80}, # ok
                    'subdural': {'L': 75, 'W': 215},
                    'bony': {'L': 600, 'W': 2800}, # ok
                    'tissue': {'L': 40, 'W': 375},
                    'stroke1': {'L': 32, 'W': 8},
                    'stroke2': {'L': 40, 'W': 40} # ok
                }

    def set_datasource(self, datasource):
        # need to call by sampler to set new datasource
        self.datasource = datasource

    def _load_img(self, img_path):
        try:
            # read gray scale img as 3 channel
            img = cv2.imread(img_path.as_posix(), cv2.IMREAD_GRAYSCALE)
        except:
            print('Missing file', img_path)
        return img

    def _load_img_file(self, img_file):
        imgs = []
        if self.mode in ['train', 'val']:
            base_dir = STAGE_1_TRAIN_DIR
        else:
            base_dir = STAGE_2_TEST_DIR
            # base_dir = STAGE_1_TEST_DIR

        for window in self.ct_windows:
            window_variant = base_dir % (self.ct_windows[window]['L'], self.ct_windows[window]['W'])
            img_path = Path(self.args.data_dir)/window_variant/f'{img_file}.jpg'
            img = self._load_img(img_path)
            imgs.append(img)
        img = np.stack(imgs, axis=-1)

        if self.mix_window == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB) # single brain window

        return img

    def get_slice(self, info):
        '''
        return tensor (c, h, w)
        c: number of windows to use, 1, 3 or 6
        '''
        # img_file = info['image']
        # img_name = img_file.split('.')[0]
        img_name = info['image']

        if self.mode in ['train', 'val']:
            label = [int(v) for v in info[self.class_name].values]
            label = Tensor(label)

        if self.mode == 'train':
            img = self._load_img_file(img_name)
            img = self.aug(img)
            return img, label
        elif self.mode == 'val':
            img = self._load_img_file(img_name)
            img = self.aug(img)
            return img, label
        else:
            img = self._load_img_file(img_name)
            return img, img_name


    def get_study(self, info):
        '''
        return tensor (s, c, h, w)
        s: # of slices per study, default 10
        c: number of windows to use, 1, 3 or 6
        '''
        study_name, study_df = info
        img_names = study_df['image'].values

        if self.mode in ['train', 'val']:
            labels = study_df[self.class_name].values
            labels = Tensor(labels)

        # for training, get args.nslices images
        if self.mode == 'train':
            l = len(img_names)
            if np.random.uniform() < 0.5:
                idx = contiguous_slice_sampling(l, self.args.nslices)
            else:
                idx = random_range_uniform_sorted(l, self.args.nslices)
            # idx = random_range_uniform_unsorted(l, self.args.nslices)

            if np.random.uniform(0) < 0.5:
                # temporal flip
                idx = idx[::-1] - np.zeros_like(idx)

            ## choose sub set
            img_names = img_names[idx]
            labels = labels[idx]
            

        if self.mode == 'train':
            # random pick n image
            imgs = [self._load_img_file(img_name) for img_name in img_names]
            imgs = [self.aug(img) for img in imgs]
            imgs =  torch.stack(imgs)
            return imgs, labels
        elif self.mode == 'val':
            
            # print('Original image')
            # print(idx)
            # print()
            # if args.temporal_flip:
            #     # add reverse idx
            #     reverse_idx = idx[::-1] - np.zeros_like(idx)
            #     all_idx = idx + reverse_idx
                
            #     # add reverse id image and label
            #     img_names = img_names + img_names[reverse_idx]
            #     labels = labels + labels[reverse_idx]

            imgs = [self._load_img_file(img_name) for img_name in img_names]
            # imgs = [self.aug(img) for img in imgs]
            # imgs = torch.stack(imgs)

            if self.args.inference:
                # for writing to csv
                return self.tta(imgs), img_names
            else:
                # for cal val loss
                return self.tta(imgs), labels
        elif self.mode == 'test':
            # load image
            imgs = [self._load_img_file(img_name) for img_name in img_names]
            return self.tta(imgs), img_names
    
    ################################################
    # run tta, only run in val and test mode
    ################################################
    def tta(self, imgs):
        # input: imgs list([h, w, c]), len=nslices
        # outputs: imgs: tensor(tta, nslices, c, h, w)
        if self.args.tta == 1:
            # original image
            resized_imgs = [cv2.resize(img, (self.args.image_size, self.args.image_size), cv2.INTER_LINEAR) for img in imgs]
            normalized_imgs = [self.aug(img) for img in resized_imgs]
            stacked_imgs = torch.stack(normalized_imgs)
            return stacked_imgs.unsqueeze_(0)
        
        elif self.args.tta == 2:
            resized_imgs = [cv2.resize(img, (self.args.image_size, self.args.image_size), cv2.INTER_LINEAR) for img in imgs]
            normalized_imgs = [self.aug(img) for img in resized_imgs]
            stacked_imgs = torch.stack(normalized_imgs)

            stacked_flip_imgs = self.fliplr(resized_imgs)
            return torch.stack((stacked_imgs, stacked_flip_imgs))

        elif self.args.tta == 3:
            resized_imgs = [cv2.resize(img, (self.args.image_size, self.args.image_size), cv2.INTER_LINEAR) for img in imgs]
            normalized_imgs = [self.aug(img) for img in resized_imgs]
            stacked_imgs = torch.stack(normalized_imgs)

            stacked_flip_lr_imgs = self.fliplr(resized_imgs)
            stacked_flip_ud_imgs = self.flipud(resized_imgs)

            return torch.stack((stacked_imgs, stacked_flip_lr_imgs, stacked_flip_ud_imgs))

        elif self.args.tta == 5 or self.args.tta == 10:
            resized_size = self.args.image_size + 32
            resized_imgs = [cv2.resize(img, (resized_size, resized_size), cv2.INTER_LINEAR) for img in imgs]

            ### Top left corner
            tl_imgs = [img[0:self.args.image_size, 0:self.args.image_size,...] for img in resized_imgs]
            normalized_tl_imgs = [self.aug(img) for img in tl_imgs]
            stacked_tl_imgs = torch.stack(normalized_tl_imgs)

            ### Top right corner
            tr_imgs = [img[0:self.args.image_size, 32:resized_size,...] for img in resized_imgs]
            normalized_tr_imgs = [self.aug(img) for img in tr_imgs]
            stacked_tr_imgs = torch.stack(normalized_tr_imgs)

            ### Bottom left
            bl_imgs = [img[32:resized_size, 0:self.args.image_size,...] for img in resized_imgs]
            normalized_bl_imgs = [self.aug(img) for img in bl_imgs]
            stacked_bl_imgs = torch.stack(normalized_bl_imgs)
            
            ### Bottom right
            br_imgs = [img[32:resized_size, 32:resized_size,...] for img in resized_imgs]
            normalized_br_imgs = [self.aug(img) for img in br_imgs]
            stacked_br_imgs = torch.stack(normalized_br_imgs)

            ### Center
            center_imgs = [img[16:resized_size-16, 16: resized_size-16,...] for img in resized_imgs]
            normalized_center_imgs = [self.aug(img) for img in center_imgs]
            stacked_center_imgs = torch.stack(normalized_center_imgs)
            
            if self.args.tta == 5:
                return torch.stack((stacked_tl_imgs, stacked_tr_imgs, stacked_bl_imgs, stacked_br_imgs, stacked_center_imgs))
            else:
                stacked_flip_tl_imgs = self.fliplr(tl_imgs)
                stacked_flip_tr_imgs = self.fliplr(tr_imgs)
                stacked_flip_bl_imgs = self.fliplr(bl_imgs)
                stacked_flip_br_imgs = self.fliplr(br_imgs)
                stacked_flip_center_imgs = self.fliplr(center_imgs)
                return torch.stack((stacked_tl_imgs, stacked_tr_imgs, stacked_bl_imgs, stacked_br_imgs, stacked_center_imgs,
                                    stacked_flip_tl_imgs, stacked_flip_tr_imgs, stacked_flip_bl_imgs, stacked_flip_br_imgs, stacked_flip_center_imgs,))
        
        raise ValueError("Unknow tta")
    
    def fliplr(self, imgs):
        flip_imgs = [np.fliplr(img) for img in imgs]
        normalized_imgs = [self.aug(img) for img in flip_imgs]
        stacked_flip_imgs = torch.stack(normalized_imgs)
        return stacked_flip_imgs

    def flipud(self, imgs):
        flip_imgs = [np.flipud(img) for img in imgs]
        normalized_imgs = [self.aug(img) for img in flip_imgs]
        stacked_flip_imgs = torch.stack(normalized_imgs)
        return stacked_flip_imgs

    def __getitem__(self, index):

        if self.args.input_level == 'per-slice':
            # datasource is slices_df
            info = self.datasource.iloc[index]
            return self.get_slice(info)
        elif self.args.input_level == 'per-study':
            # datasource is studies_df
            info = self.datasource[index]
            return self.get_study(info)


    def __len__(self):
        if self.mode == 'train':
            # Get len from sampler instead
            return 0
        return len(self.datasource)

def get_train_dl(args):
    ds = ImagesDS(args, HarderAugmenter(args), mode='train')

    fold_path = Path(args.data_dir)/'fold'

    # train by studies or patients
    df = pd.read_csv(fold_path/f"trainset_stage1_split_patients.csv")
    train_df = df.loc[df['fold'] != args.fold]

    # sampler will set datasource
    sampler = BalanceClassSampler(train_df, ds, args)
    return D.DataLoader(
        ds,
        sampler=sampler,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        pin_memory=True)


def get_val_dl(args):
    ds = ImagesDS(args, ValAugmenter(args), mode='val')

    # set datasource for dataset
    fold_path = Path(args.data_dir)/'fold'
    df = pd.read_csv(fold_path/f"trainset_stage1_split_patients.csv")

    df = df[df.BrainPresence]

    val_df = df.loc[df['fold'] == args.fold]
    if args.input_level == 'per-slice':
        slices_df = val_df
        ds.set_datasource(slices_df)
        print('Val dataset #slices', slices_df.shape)
    elif args.input_level == 'per-study':
        studies_df = val_df.groupby('StudyInstanceUID')
        list_studies_df = list(studies_df)
        ds.set_datasource(list_studies_df)
        print('Val dataset #studies', len(list_studies_df))

    # construct dataloader
    # bs = 1 if args.input_level == 'per-study' else args.batch_size*2
    # if args.inference:
    collate_fn = id_collate if args.inference else None # collect label for inference
    return D.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True)


from torch.utils.data.dataloader import default_collate


def id_collate(batch):
    new_batch = []
    ids = []
    for _batch in batch:
        new_batch.append(_batch[:-1])
        ids.append(_batch[-1])
    return default_collate(new_batch), ids

def get_test_dl(args, df=None):
    ds = ImagesDS(args, TestAugmenter(args), mode='test')

    if df is None:
        # df = pd.read_csv(Path(args.data_dir)/'fold/testset_stage1.csv')
        df = pd.read_csv(Path(args.data_dir)/'fold/testset_stage2.csv')
        # IMPORTANT: we dont make prediction on image w/o brain
        df = df[df.BrainPresence]

    if args.input_level == 'per-slice':
        slices_df = df
        ds.set_datasource(slices_df)
        print('Test dataset #slices', slices_df.shape)
    elif args.input_level == 'per-study':
        studies_df = df.groupby('StudyInstanceUID')
        list_studies_df = list(studies_df)
        ds.set_datasource(list_studies_df)
        print('Test dataset #studies', len(list_studies_df))

    # construct dataloader
    # bs = 1 if args.input_level == 'per-study' else args.batch_size*2
    return D.DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.workers,
        drop_last=False,
        collate_fn=id_collate,
        pin_memory=True)


if __name__=='__main__':
    print("Test")


from torch.utils.data.sampler import Sampler, SequentialSampler, RandomSampler


# https://github.com/bestfitting/kaggle/blob/master/siim_acr/src/datasets/datasets.py
class BalanceClassSampler(Sampler):
    """
    Balance each class in sample data
    """
    def __init__(self, df, dataset, args):
        self.dataset = dataset
        self.df = df
        self.args = args

        # IMPORTANT: we dont train on image w/o brain (to save time)
        self.df = self.df[self.df.BrainPresence]

        if args.input_level == 'per-slice':
            self.sample_per_slice()
        elif args.input_level == 'per-study':
            self.sample_per_study()

        self.info()

    def info(self):
        if self.args.input_level == 'per-slice':
            print('='*10, 'Train per slice', '='*10)
            print(f'pos slices: {self.nrof_pos}, neg slices: {self.nrof_neg}')
            print(f'epidural: {self.nrof_epidural}, subarachnoid: {self.nrof_subarachnoid}, subdural: {self.nrof_subdural}, intraparenchymal: {self.nrof_intraparenchymal}, intraventricular: {self.nrof_intraventricular}')
        elif self.args.input_level == 'per-study':
            print('='*10, 'Train per study', '='*10)
            print(f'pos studies: {self.nrof_pos}, neg studies: {self.nrof_neg}')

    #########################################################
    # Sample at slice level,  length = total number of slice
    #########################################################
    def sample_per_slice(self):
        self.slices_df = self.df

        self.nrof_epidural = self.df[self.df.epidural == 1].count()[0]
        self.nrof_subarachnoid = self.df[self.df.subarachnoid == 1].count()[0]
        self.nrof_subdural = self.df[self.df.subdural == 1].count()[0]
        self.nrof_intraparenchymal = self.df[self.df.intraparenchymal == 1].count()[0]
        self.nrof_intraventricular = self.df[self.df.intraventricular == 1].count()[0]
        self.nrof_pos = self.nrof_epidural + self.nrof_subarachnoid + self.nrof_subdural + self.nrof_intraparenchymal + self.nrof_intraventricular
        self.nrof_neg = self.df[self.df['any']==0].count()[0]
        self.length = self.nrof_neg + self.nrof_pos

        self.dataset.set_datasource(self.slices_df)


    def iter_slice_index(self):
        # return random slice indexes to train
        epidural_index = np.where(self.df.epidural==1)[0]
        subarachnoid_index = np.where(self.df.subarachnoid==1)[0]
        subdural_index = np.where(self.df.subdural==1)[0]
        intraparenchymal_index = np.where(self.df.intraparenchymal==1)[0]
        intraventricular_index = np.where(self.df.intraventricular==1)[0]
        neg_index = np.where(self.df['any']==0)[0]

        epidural = np.random.choice(epidural_index, self.nrof_epidural, replace=True)
        subarachnoid = np.random.choice(subarachnoid_index, self.nrof_subarachnoid, replace=True)
        subdural = np.random.choice(subdural_index, self.nrof_subdural, replace=True)
        intraparenchymal = np.random.choice(intraparenchymal_index, self.nrof_intraparenchymal, replace=True)
        intraventricular = np.random.choice(intraventricular_index, self.nrof_intraventricular, replace=True)
        neg = np.random.choice(neg_index, self.nrof_neg, replace=True)

        l = np.hstack([epidural, subarachnoid, subdural, intraparenchymal, intraventricular, neg]).T
        l = l.reshape(-1)
        np.random.shuffle(l)
        l = l[:self.length]
        return iter(l)


    #########################################################
    # Sample at study level, length = total number of studies
    #########################################################
    def sample_per_study(self):
        studies_df = self.df.groupby('StudyInstanceUID')
        any_df = studies_df.agg({'any': 'sum'})

        self.list_studies_df = list(studies_df)
        self.nrof_pos = (any_df['any'] > 0).sum()
        self.nrof_neg = (any_df['any'] == 0).sum()
        self.length = self.nrof_pos + self.nrof_neg

        self.dataset.set_datasource(self.list_studies_df)

    def iter_study_index(self):
        # return random study indexes to train
        l = np.arange(self.length)
        np.random.shuffle(l)
        return iter(l)


    #########################################################
    # # TODO: Gradually remove easy and no-brain image
    #########################################################
    # def step(self, e):
    #     # step per epoch e
    #     self.epoch = e
    #     self.cheery_pick()

    #     if self.args.input_level == 'per-slice':
    #         self.dataset.set_datasource(self.slices_df)
    #     elif self.args.input_level == 'per-study':
    #         self.dataset.set_datasourcer(self.studies_df)

    def cheery_pick(self):
        print('NO CHERRY PICK.')
        return

    #########################################################
    # Framework method
    #########################################################
    def __iter__(self):
        # get index for each disease
        if self.args.input_level == 'per-slice':
            return self.iter_slice_index()
        elif self.args.input_level == 'per-study':
            return self.iter_study_index()

    def __len__(self):
        # Return dataloader length
        return self.length



