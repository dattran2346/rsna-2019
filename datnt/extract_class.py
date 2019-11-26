import torch
from collections import OrderedDict
import glob
import os
import argparse

def extract_class(cp):
    cp_dict = torch.load(cp, 'cpu')
    for key in ['resnet.fc.weight', 'resnet.fc.bias']:
        cp_dict[key] = cp_dict[key][0].unsqueeze(0)
    new_state_dict = OrderedDict()
    for k, v in cp_dict.items():
        new_state_dict[k] = v

    torch.save(new_state_dict, cp.replace('.pth', '_any.pth'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--cp', type=str)
    cp = parser.parse_args().cp

    extract_class(cp)
