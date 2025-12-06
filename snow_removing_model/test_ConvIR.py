import os
import torch
from pytorch_msssim import ssim
from torchvision.transforms import functional
from utils import Adder
from data import test_dataloader
from skimage.metrics import peak_signal_noise_ratio
import torch.nn.functional as F
from model.ConvIR import ConvIR
import argparse

parser = argparse.ArgumentParser()

# Directories
parser.add_argument('--data_dir', type=str, default='datasets/DesnowDataset')

# Test
parser.add_argument('--test_model', type=str, default='snow_removing_model/weights/best.pkl')
parser.add_argument('--save_image', type=bool, default=True, choices=[True, False])

args = parser.parse_args()
args.result_dir = os.path.join('snow_removing_model', 'results')
if not os.path.exists(args.result_dir):
    os.makedirs(args.result_dir)


def test(model):
    state_dict = torch.load(args.test_model)
    model.load_state_dict(state_dict['model'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = test_dataloader(args.data_dir, batch_size=1, num_workers=0)
    torch.cuda.empty_cache()
    model.eval()
    factor = 32
    with torch.no_grad():
        psnr_adder = Adder()
        ssim_adder = Adder()

        for iter_idx, data in enumerate(dataloader):
            input_img, label_img, name = data
            input_img = input_img.to(device)

            h, w = input_img.shape[2], input_img.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor*factor)
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            input_img = F.pad(input_img, (0, padw, 0, padh), 'reflect')

            pred = model(input_img)[2]
            pred = pred[:,:,:h,:w]

            pred_clip = torch.clamp(pred, 0, 1)

            pred_numpy = pred_clip.squeeze(0).cpu().numpy()
            label_numpy = label_img.squeeze(0).cpu().numpy()


            if args.save_image:
                save_name = os.path.join(args.result_dir, name[0])
                pred_clip += 0.5 / 255
                pred = functional.to_pil_image(pred_clip.squeeze(0).cpu(), 'RGB')
                pred.save(save_name)


            label_img = (label_img).cuda()
            down_ratio = max(1, round(min(H, W) / 256))
            ssim_val = ssim(F.adaptive_avg_pool2d(pred_clip, (int(H / down_ratio), int(W / down_ratio))), 
                            F.adaptive_avg_pool2d(label_img, (int(H / down_ratio), int(W / down_ratio))), 
                            data_range=1, size_average=False)	
            ssim_adder(ssim_val)
           
            psnr = peak_signal_noise_ratio(pred_numpy, label_numpy, data_range=1)
            psnr_adder(psnr)

            print('%d iter PSNR: %.2f SSIM: %f' % (iter_idx + 1, psnr, ssim_val))

        print('==========================================================')
        print('The average PSNR is %.2f dB' % (psnr_adder.average()))
        print('The average SSIM is %.4f' % (ssim_adder.average()))

if __name__ == '__main__':
    model = ConvIR()
    if torch.cuda.is_available():
        model.cuda()

    test(model)