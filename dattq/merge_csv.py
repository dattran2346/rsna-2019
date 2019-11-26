import pandas as pd
from pathlib import Path
import argparse
import numpy as np

parser = argparse.ArgumentParser(
            description='Merge csv from nhannt')

parser.add_argument('--train-metadata', default=None, help='nhannt train metatdata')
parser.add_argument('--val-fold-dir', default=None, help='Val split fold dir')
parser.add_argument('--output-dir', default=None, help='Place to save train csv')

args = parser.parse_args()

# train_df = pd.read_csv(input_dir/'train_metadata.csv')
train_df = pd.read_csv(args.train_metadata)

input_dir = Path(args.val_fold_dir)
val_fold0 = np.load(input_dir/'valid_fold0.npy', allow_pickle=True)
val_fold1 = np.load(input_dir/'valid_fold1.npy', allow_pickle=True)
val_fold2 = np.load(input_dir/'valid_fold2.npy', allow_pickle=True)
val_fold3 = np.load(input_dir/'valid_fold3.npy', allow_pickle=True)
val_fold4 = np.load(input_dir/'valid_fold4.npy', allow_pickle=True)

## Split by patient ids
val_fold0 = np.load(input_dir/'valid_fold0.npy', allow_pickle=True)
val_fold1 = np.load(input_dir/'valid_fold1.npy', allow_pickle=True)
val_fold2 = np.load(input_dir/'valid_fold2.npy', allow_pickle=True)
val_fold3 = np.load(input_dir/'valid_fold3.npy', allow_pickle=True)
val_fold4 = np.load(input_dir/'valid_fold4.npy', allow_pickle=True)

# save to output dir
output_dir = Path(args.output_dir)
output_dir.mkdir(exist_ok=True, parents=True)

train_df.to_csv(output_dir/'trainset_stage1_split_patients.csv', index=False)
