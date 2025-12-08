# This code defines a dataset class and data preparation functions for training a rain removal model.

import os
import numpy as np
import torch
import cv2
import torch.utils.data as udata
from utils import *


class Dataset(udata.Dataset):
    def __init__(self, data_path='.', img_size=(100, 100)):
        super(Dataset, self).__init__()
        self.data_path = data_path
        self.input_dir = os.path.join(self.data_path, 'rainy_image')
        self.target_dir = os.path.join(self.data_path, 'ground_truth')
        self.img_exts = ['.jpg', '.png', '.jpeg', '.bmp']
        self.input_files = [f for f in os.listdir(self.input_dir) if any(f.lower().endswith(ext) for ext in self.img_exts)]
        self.input_files.sort()
        self.target_files = [f for f in os.listdir(self.target_dir) if any(f.lower().endswith(ext) for ext in self.img_exts)]
        self.target_files.sort()
        # ground_truth的文件名为rain_train_1.jpg，rainy_image的文件名为rain_train_1_1.jpg
        self.pairs = [(os.path.join(self.input_dir, f), os.path.join(self.target_dir, '_'.join(f.split('_')[:-1]) + '.jpg')) 
                      for f in self.input_files if '_'.join(f.split('_')[:-1]) + '.jpg' in self.target_files]
        self.size = img_size

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)
        # BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        # Resize
        input_img = cv2.resize(input_img, self.size, interpolation=cv2.INTER_AREA)
        target_img = cv2.resize(target_img, self.size, interpolation=cv2.INTER_AREA)
        # 转为float32并归一化到[0,1]
        input_img = input_img.astype(np.float32) / 255.0
        target_img = target_img.astype(np.float32) / 255.0
        # HWC to CHW
        input_img = np.transpose(input_img, (2, 0, 1))
        target_img = np.transpose(target_img, (2, 0, 1))
        return torch.Tensor(input_img), torch.Tensor(target_img)