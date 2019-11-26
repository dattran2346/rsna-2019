# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
from .image_list import to_image_list


class BatchCollator(object):
    """
    From a list of samples from the dataset,
    returns the batched images and targets.
    This should be passed to the DataLoader
    """

    def __init__(self, size_divisible=0):
        self.size_divisible = size_divisible

    def __call__(self, batch):
        transposed_batch = list(zip(*batch))
     
        images = to_image_list(transposed_batch[0], self.size_divisible)
        
        # multi-task
        if len(transposed_batch) == 3:
            targets_0 = torch.stack(transposed_batch[1], dim=0)
            targets_1 = torch.stack(transposed_batch[2], dim=0)
            return images.tensors, targets_0, targets_1
        
        if isinstance(transposed_batch[1][0], str):
            return images.tensors, transposed_batch[1]
            
        targets = torch.stack(transposed_batch[1], dim=0)
        return images.tensors, targets