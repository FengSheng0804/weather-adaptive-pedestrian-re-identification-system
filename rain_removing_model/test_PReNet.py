# This code is used to test a rain removal model using PReNet.
import cv2
import os
import argparse
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from model.PReNet import *
from data.DerainDataset import test_dataloader
import time

parser = argparse.ArgumentParser(description="PReNet_Test")
parser.add_argument("--logdir", type=str, default="rain_removing_model/weights", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="datasets/DerainDataset/train", help='path to training data')
parser.add_argument("--result_dir", type=str, default="rain_removing_model/results", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

opt.predict_result_dir = os.path.join(opt.result_dir, 'predict')
if not os.path.exists(opt.predict_result_dir):
    os.makedirs(opt.predict_result_dir)

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def test():
    
    os.makedirs(opt.predict_result_dir, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = PReNet(opt.recurrent_iter, opt.use_GPU)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'best.pth')))
    model.eval()

    time_test = 0
    count = 0

    # test的时候需要将batch_size设置为1，以便处理不同尺寸的图片
    loader = test_dataloader(opt.data_path, batch_size=1, num_workers=0)
    for img_tensor, img_name in loader:
        img = img_tensor
        if opt.use_GPU:
            img = img.cuda()
        with torch.no_grad():
            if opt.use_GPU:
                torch.cuda.synchronize()
            start_time = time.time()
            out, _ = model(img)
            out = torch.clamp(out, 0., 1.)
            if opt.use_GPU:
                torch.cuda.synchronize()
            end_time = time.time()
            dur_time = end_time - start_time
            time_test += dur_time
            print(img_name[0], ': ', dur_time)
        save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())
        save_out = save_out.transpose(1, 2, 0)
        b, g, r = cv2.split(save_out)
        save_out = cv2.merge([r, g, b])

        # 不再resize，直接保存原图大小
        cv2.imwrite(os.path.join(opt.predict_result_dir, img_name[0]), save_out)
        count += 1
    print('Avg. time:', time_test/count)


if __name__ == "__main__":
    test()