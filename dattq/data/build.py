# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy

import torch.utils.data
import torchvision.transforms as transforms

from .datasets import ImetDataset
from .collate_batch import BatchCollator
from .transform import Lighting, Bound
from . import samplers


def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler


def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized


def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios


def make_batch_data_sampler(
    dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0
):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(
            sampler, group_ids, images_per_batch, drop_uneven=False
        )
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(
            sampler, images_per_batch, drop_last=False
        )
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(
            batch_sampler, num_iters, start_iter
        )
    return batch_sampler


def make_data_loader(cfg, is_distributed=False, start_iter=0):
    to_tensor_and_normalize = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize(cfg['MEAN'], cfg['STD'])
        ])
    
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(cfg['TRAIN_SIZE'], interpolation=cfg['INTERPOLATION']),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2),
        transforms.ToTensor(),
        Lighting(),
        Bound(0., 1.),
        transforms.Normalize(cfg['MEAN'], cfg['STD'])
        ])
    val_transforms = transforms.Compose([
        transforms.Resize(cfg['TEST_SIZE'], interpolation=cfg['INTERPOLATION']),
        transforms.CenterCrop(cfg['TRAIN_SIZE']),
        to_tensor_and_normalize
        ])
    test_transforms = transforms.Compose([
        transforms.Resize(cfg['TEST_SIZE'], interpolation=cfg['INTERPOLATION']),
        transforms.CenterCrop(cfg['TRAIN_SIZE']),
        to_tensor_and_normalize
        ])

    datasets = [
        (ImetDataset(cfg["DATA_DIR"], cfg['TRAIN_FILE'], "train", train_transforms), True),
        (ImetDataset(cfg["DATA_DIR"], cfg['VALID_FILE'], "val", val_transforms), False),
        (ImetDataset(cfg["DATA_DIR"], "sample_submission.csv", "test", test_transforms), False)
    ]

    data_loaders = []
    for tupl in datasets:
        dataset, shuffle = tupl
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(
            dataset, sampler,
            cfg['ASPECT_GROUPING'], cfg['IMAGES_PER_BATCH'], 
            cfg['NUM_ITERS']
        )
        collator = BatchCollator()
        num_workers = cfg['NUM_WORKERS']
        data_loader = torch.utils.data.DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=batch_sampler,
            collate_fn=collator,
        )
        data_loaders.append(data_loader)
    
    return data_loaders
