# 使用之前，需要先使用reparam.py将train_DEANet.py训练出来的模型转换为测试模式

import os
import torch
import cv2
import numpy as np
from torchvision.utils import save_image
from tqdm import tqdm
import time

# Assuming the necessary modules are in the same directory or accessible in the python path.
# If not, you might need to adjust the sys.path
import sys
# Add the parent directory to the path to allow imports from sibling directories
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from fog_removing_model.model.backbone_train import DEANet
from model.backbone import Backbone
from utils import pad_img
import argparse


parser = argparse.ArgumentParser(description='DEANet Testing')
parser.add_argument('--weights_path', type=str, 
                    default='fog_removing_model\\weights\\best.pth',
                    help='Path to the trained weights file')
parser.add_argument('--data_path', type=str, 
                    default='datasets\\DefogDataset\\test',
                    help='Path to the test data directory')
parser.add_argument('--save_path', type=str, 
                    default='fog_removing_model\\results',
                    help='Path to save the de-fogged images')
parser.add_argument('--use_GPU', action='store_true', help='Use GPU for testing')
parser.add_argument('--gpu_id', type=str, default="0", help='GPU ID to use')

opt = parser.parse_args()
# The original logic was to use GPU if available. We can replicate that.
if not opt.use_GPU:
    opt.use_GPU = torch.cuda.is_available()


def is_image(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.bmp'])

def normalize(img):
    return img / 255.0

def test():
    if opt.use_GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = Backbone()
    if opt.use_GPU:
        model = model.cuda()
    
    # Load weights
    try:
        model.load_state_dict(torch.load(opt.weights_path))
    except RuntimeError:
        # This may happen if the model was saved with DataParallel
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        state_dict = torch.load(opt.weights_path)
        for k, v in state_dict.items():
            name = k[7:] # remove `module.`
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)

    model.eval()

    time_test = 0
    count = 0
    foggy_image_dir = os.path.join(opt.data_path, 'foggy_image')

    image_files = [f for f in os.listdir(foggy_image_dir) if is_image(f)]

    for img_name in tqdm(image_files, desc='Testing'):
        img_path = os.path.join(foggy_image_dir, img_name)

        # input image
        hazy_img = cv2.imread(img_path)
        hazy_img = cv2.cvtColor(hazy_img, cv2.COLOR_BGR2RGB) # convert to RGB
        
        hazy_img_normalized = normalize(np.float32(hazy_img))
        hazy_tensor = torch.Tensor(hazy_img_normalized).permute(2, 0, 1).unsqueeze(0)

        if opt.use_GPU:
            hazy_tensor = hazy_tensor.cuda()

        with torch.no_grad():
            H, W = hazy_tensor.shape[2:]
            hazy_padded = pad_img(hazy_tensor, 4)

            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()

            output = model(hazy_padded)
            output = output.clamp(0, 1)
            output = output[:, :, :H, :W]

            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time

            # print(img_name, ': ', dur_time)

        save_path = os.path.join(opt.save_path, img_name)
        save_image(output, save_path)

        count += 1

    print(f'\nTesting finished. Results saved to {opt.save_path}')
    if count > 0:
        print(f'Avg. time: {time_test/count:.4f} seconds')

if __name__ == "__main__":
    test()