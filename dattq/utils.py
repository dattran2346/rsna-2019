from collections import OrderedDict
import shutil
import torch
from pathlib import Path
import random
import numpy as np

save_dir = Path('runs')

def save_checkpoint(state, is_best, checkname, epoch, save_all=False):
    print('Save checkpoint', checkname)
    check_dir = save_dir/checkname
    check_dir.mkdir(exist_ok=True)

    if save_all:
        checkpoint_file = f'checkpoint_{epoch}.pth'
    else:
        checkpoint_file = 'checkpoint.pth'
    torch.save(state, check_dir/checkpoint_file)

    if is_best:
        shutil.copyfile(check_dir/checkpoint_file, check_dir/'best_model.pth')


def load_state_dict(state_dict, model):
    new_state_dict = OrderedDict()
    for key, value in state_dict.items():
        if key.startswith('module.'):
            print('Modify state dict')
            new_key = key[7:]
        else:
            new_key = key
        new_state_dict[new_key] = value
    model.load_state_dict(new_state_dict)


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

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
