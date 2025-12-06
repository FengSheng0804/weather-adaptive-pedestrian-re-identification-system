# This code defines a dataset class and data preparation functions for training a rain removal model.

import os
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import *


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)

    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def prepare_data(data_path, patch_size, stride):
    # train
    print('process training data')
    input_path = os.path.join(data_path, 'rainy_image')
    target_path = os.path.join(data_path, 'ground_truth')

    save_target_path = os.path.join(data_path, 'train_target.h5')
    save_input_path = os.path.join(data_path, 'train_input.h5')

    target_h5f = h5py.File(save_target_path, 'w')
    input_h5f = h5py.File(save_input_path, 'w')

    train_num = 0
    input_files = glob.glob(os.path.join(input_path, '*.jpg'))
    target_files = glob.glob(os.path.join(target_path, '*.jpg'))

    for i in range(len(target_files)):
        target = cv2.imread(os.path.normpath(target_files[i]))
        b, g, r = cv2.split(target)
        target = cv2.merge([r, g, b])

        input = cv2.imread(os.path.normpath(input_files[i]))
        b, g, r = cv2.split(input)
        input = cv2.merge([r, g, b])

        target_patches = Im2Patch(target, win=patch_size, stride=stride)
        input_patches = Im2Patch(input, win=patch_size, stride=stride)

        target_img = target
        target_img = np.float32(normalize(target_img))
        input_img = input
        input_img = np.float32(normalize(input_img))

        for n in range(target_patches.shape[3]):
            target_data = target_patches[:, :, :, n].copy()
            target_h5f.create_dataset(str(train_num), data=target_data)

            input_data = input_patches[:, :, :, n].copy()
            input_h5f.create_dataset(str(train_num), data=input_data)
            train_num += 1

    target_h5f.close()
    input_h5f.close()
    print('training set, # samples %d\n' % train_num)


class Dataset(udata.Dataset):
    def __init__(self, data_path='.'):
        super(Dataset, self).__init__()

        self.data_path = data_path

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        self.keys = list(target_h5f.keys())
        random.shuffle(self.keys)
        target_h5f.close()
        input_h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):

        target_path = os.path.join(self.data_path, 'train_target.h5')
        input_path = os.path.join(self.data_path, 'train_input.h5')

        target_h5f = h5py.File(target_path, 'r')
        input_h5f = h5py.File(input_path, 'r')

        key = self.keys[index]
        target = np.array(target_h5f[key])
        input = np.array(input_h5f[key])

        target_h5f.close()
        input_h5f.close()

        return torch.Tensor(input), torch.Tensor(target)