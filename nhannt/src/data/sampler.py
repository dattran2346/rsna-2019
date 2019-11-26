import numpy as np 
import torch 
import torch.distributed as dist
from torch.utils.data import Sampler


class DistributedSampler(Sampler):
    """
    Sampler that restricts data loading to a subset of the dataset.
    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such case, each
    process can pass a DistributedSampler instance as a DataLoader sampler,
    and load a subset of the original dataset that is exclusive to it.
    .. note::
        Dataset is assumed to be of constant size.
    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = torch.arange(len(self.dataset)).tolist()

        # add extra samples to make it evenly divisible
        indices += indices[: (self.total_size - len(indices))]
        assert len(indices) == self.total_size

        # subsample
        offset = self.num_samples * self.rank
        indices = indices[offset: offset + self.num_samples]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class EasySampler(Sampler):
    """Samples easy elements randomly from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, hard_indices, easy_indices, ratio=0.2):
        self.hard_indices = hard_indices
        self.easy_indices = easy_indices
        self.ratio = ratio
        self.length = len(hard_indices) + int(len(hard_indices) * ratio)

    def __iter__(self):
        sampled_easy_indices = np.random.choice(self.easy_indices,
                                                size=int(
                                                    len(self.hard_indices) * self.ratio),
                                                replace=False)
        indices = np.append(self.hard_indices, sampled_easy_indices)
        return (indices[i] for i in torch.randperm(len(indices)))

    def __len__(self):
        return self.length


def dattran_slices_sampler(nslices, l):
    """
    Arguments:
        nslices (int): number of slices sampled from a study
        l (int): length of a study
    """
    mid = np.random.randint(0, l)
    start = mid - nslices//2
    end = mid + (nslices//2 + nslices%2)
    idx = np.arange(start, end)
    idx += (1 - min(0, idx[0]))
    idx -= ((max(l, idx[-1]) - l)+1)
    return idx 


def random_sorted_slices_sampler(nslices, l):
    """
    Arguments:
        nslices (int): number of slices sampled from a study
        l (int): length of a study
    """
    idx = np.random.choice(range(l), nslices, replace=False)
    idx = sorted(idx)
    return idx 