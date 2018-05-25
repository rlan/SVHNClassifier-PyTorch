"""
Find mean and std of the dataset

https://pytorch.org/docs/stable/torchvision/transforms.html

"""

from __future__ import division
from __future__ import print_function

import argparse
import os
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn.functional
import torch.utils.data
from dataset import DatasetRaw

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--data_dir', default='./data', help='directory to read LMDB files')
parser.add_argument('-b', '--batch_size', default=32, type=int,  help='Default 32')

def _train(path_to_train_lmdb_dir,
           path_to_val_lmdb_dir,
           training_options):
    batch_size = training_options['batch_size']

    print("Loading data...")
    train_loader = torch.utils.data.DataLoader(DatasetRaw(path_to_train_lmdb_dir),
                                               num_workers=2,
                                               batch_size=batch_size)
    print("Done")

    start_time = time.time()
    for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
        print(batch_idx, images.shape)

    duration = time.time() - start_time
    print("duration {} seconds", duration)


def main(args):
    path_to_train_lmdb_dir = os.path.join(args.data_dir, 'train.lmdb')
    path_to_val_lmdb_dir = os.path.join(args.data_dir, 'val.lmdb')
    training_options = {
        'batch_size': args.batch_size,
    }

    print('Finding mean and std...')
    _train(path_to_train_lmdb_dir, path_to_val_lmdb_dir, training_options)
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
