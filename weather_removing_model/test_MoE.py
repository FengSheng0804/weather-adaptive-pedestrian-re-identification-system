import os
import torch
import argparse
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.moe import MoE
from data.data_loader import MoETestDataset
from utils.SSIM import ssim, psnr

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = MoE(args.class_num)
    # map_location='cpu' 防止显存不足时加载失败
    checkpoint = torch.load(args.weights_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 加载数据
    test_set = MoETestDataset(root_dir=args.data_dir, crop_size=256, use_transform=False, scenario='all')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    total_psnr, total_ssim, count = 0, 0, 0
    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs, targets, img_names, scores = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 输入补齐到32的倍数
            factor = 32
            h, w = inputs.shape[2], inputs.shape[3]
            H, W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
            padh = H-h if h%factor!=0 else 0
            padw = W-w if w%factor!=0 else 0
            if padh > 0 or padw > 0:
                inputs = torch.nn.functional.pad(inputs, (0, padw, 0, padh), 'reflect')
            
            # 处理score
            if isinstance(scores, dict):
                score_keys = ['fog', 'rain', 'snow']
                score_tensor = torch.stack([scores[k].to(inputs.device).to(inputs.dtype) for k in score_keys], dim=1)
            elif scores is None:
                score_tensor = None
            else:
                score_tensor = torch.tensor(scores, dtype=inputs.dtype, device=inputs.device).view(-1, 3)
            
            # 推理
            outputs = model(inputs, score=score_tensor)
            
            # 裁剪回原始尺寸
            final_out = outputs['final_output'][:, :, :h, :w].clamp(0, 1)
            moe_out = outputs['moe_output'][:, :, :h, :w].clamp(0, 1)
            expert_weights = outputs['expert_weights'].cpu().numpy()

            expert_names = ['defog', 'derain', 'desnow']
            ew = np.array(expert_weights)
            if ew.ndim == 0:
                ew = ew.reshape(1, 1)
            elif ew.ndim == 1:
                ew = ew.reshape(-1, len(expert_names))
            # 保存专家权重到txt文件
            weights_txt_path = os.path.join(args.save_dir, "expert_weights.txt")
            with open(weights_txt_path, "a") as f:
                for i in range(final_out.size(0)):
                    weights_info = ', '.join([f'{expert_names[j]}: {ew[i][j]:.4f}' for j in range(len(expert_names))])
                    f.write(f"Image: {img_names[i]}, Expert Weights: [{weights_info}]\n")

            # 计算指标
            batch_psnr = psnr(final_out, targets)
            batch_ssim = ssim(final_out, targets).item()
            total_psnr += batch_psnr * final_out.size(0)
            total_ssim += batch_ssim * final_out.size(0)
            count += final_out.size(0)
            
            # 保存图片
            for i in range(final_out.size(0)):
                # 保存最终输出
                out_path = os.path.join(args.save_dir, f"{img_names[i].split('.')[0]}_out.jpg")
                save_image(final_out[i], out_path)
                # 保存MoE输出
                moe_path = os.path.join(args.save_dir, f"{img_names[i].split('.')[0]}_moe.jpg")
                save_image(moe_out[i], moe_path)

    avg_psnr = total_psnr / count
    avg_ssim = total_ssim / count
    print(f"[Test] Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/MoEDataset', help='数据集路径')
    parser.add_argument('--weights_path', type=str, default='weather_removing_model/weights/moe_best1.pth', help='模型权重路径')
    parser.add_argument('--save_dir', type=str, default='weather_removing_model/results/predict', help='结果保存路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--class_num', type=int, default=3, help='天气类别数量')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)
