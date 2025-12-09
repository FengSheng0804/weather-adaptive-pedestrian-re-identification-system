# This code defines a dataset class and data preparation functions for training a rain removal model.

import os
import numpy as np
import torch
import cv2
from torch.utils.data import Dataset, DataLoader

class DerainPatchDataset(Dataset):
    def __init__(self, data_path, patch_size=256, use_patch=True):
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
        self.patch_size = patch_size
        self.use_patch = use_patch

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        input_path, target_path = self.pairs[index]
        input_img = cv2.imread(input_path)
        target_img = cv2.imread(target_path)
        # BGR to RGB
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)
        target_img = cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB)
        if self.use_patch:
            h, w, _ = input_img.shape
            ph, pw = self.patch_size, self.patch_size
            if h > ph and w > pw:
                top = np.random.randint(0, h - ph)
                left = np.random.randint(0, w - pw)
                input_img = input_img[top:top+ph, left:left+pw]
                target_img = target_img[top:top+ph, left:left+pw]
            else:
                input_img = cv2.resize(input_img, (pw, ph), interpolation=cv2.INTER_AREA)
                target_img = cv2.resize(target_img, (pw, ph), interpolation=cv2.INTER_AREA)
        # 转为float32并归一化到[0,1]
        input_img = input_img.astype(np.float32) / 255.0
        target_img = target_img.astype(np.float32) / 255.0
        # HWC to CHW
        input_img = np.transpose(input_img, (2, 0, 1))
        target_img = np.transpose(target_img, (2, 0, 1))
        return torch.Tensor(input_img), torch.Tensor(target_img)

class DerainFullImageDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.input_dir = os.path.join(self.data_path, 'rainy_image')
        self.img_exts = ['.jpg', '.png', '.jpeg', '.bmp']
        self.input_files = [f for f in os.listdir(self.input_dir) if any(f.lower().endswith(ext) for ext in self.img_exts)]
        self.input_files.sort()

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        img_name = self.input_files[idx]
        img_path = os.path.join(self.input_dir, img_name)
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        return torch.Tensor(img), img_name

def train_dataloader(data_path, batch_size=16, num_workers=4, patch_size=256, use_patch=True):
    dataset = DerainPatchDataset(data_path, patch_size=patch_size, use_patch=use_patch)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return loader

def test_dataloader(data_path, batch_size=1, num_workers=0):
    dataset = DerainFullImageDataset(data_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return loader