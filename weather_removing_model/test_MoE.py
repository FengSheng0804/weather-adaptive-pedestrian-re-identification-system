import os
import sys
import argparse
import torch
import numpy as np
from collections import OrderedDict
from torch.utils.data import DataLoader
from torchvision.utils import save_image

# Add current directory to sys.path to ensure imports work
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.moe import MoE
from data.data_loader import MoETestDataset
from utils.SSIM import ssim, psnr

# 去除module前缀
def strip_module_prefix(state_dict):
    """Remove a leading 'module.' in state_dict keys if present."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载模型
    model = MoE(score_dim=args.class_num)

    # 动态修复 reconstruction_net 输入通道以兼容预训练权重（权重为67通道）
    if hasattr(model, 'reconstruction_net') and model.reconstruction_net[0].in_channels != 67:
        print(f"Patching reconstruction_net input channels from {model.reconstruction_net[0].in_channels} to 67")
        model.reconstruction_net[0] = torch.nn.Conv2d(67, 64, 3, padding=1)
    
    if not os.path.exists(args.weights_path):
        print(f"Error: Weights file not found at {args.weights_path}")
        return

    print(f"Loading weights from {args.weights_path}")
    # map_location='cpu' 防止显存不足时加载失败
    checkpoint = torch.load(args.weights_path, map_location='cpu')
    state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
    state_dict = strip_module_prefix(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # 加载数据
    print(f"Loading dataset from {args.data_dir}")
    test_set = MoETestDataset(root_dir=args.data_dir, crop_size=256, use_transform=False, scenario='all')
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    total_psnr, total_ssim, count = 0, 0, 0
    
    # 清空或创建权重记录文件
    weights_txt_path = os.path.join(args.save_dir, "expert_weights.txt")
    with open(weights_txt_path, "w") as f:
        f.write("Image Name, Defog, Derain, Desnow\n")

    with torch.no_grad():
        for idx, batch in enumerate(test_loader):
            inputs, targets, img_names = batch
            inputs = inputs.to(device)
            targets = targets.to(device)

            # 输入补齐到32的倍数
            factor = 32
            h, w = inputs.shape[2], inputs.shape[3]
            # 修正padding逻辑
            padh = (factor - h % factor) % factor
            padw = (factor - w % factor) % factor
            
            if padh > 0 or padw > 0:
                inputs = torch.nn.functional.pad(inputs, (0, padw, 0, padh), 'reflect')
            
            # 推理（测试阶段不提供分数）
            outputs = model(inputs, score=None)
            
            # 裁剪回原始尺寸
            final_out = outputs['final_output'][:, :, :h, :w].clamp(0, 1)
            moe_out = outputs['moe_output'][:, :, :h, :w].clamp(0, 1)
            expert_weights = outputs['expert_weights']
            
            if isinstance(expert_weights, torch.Tensor):
                expert_weights = expert_weights.cpu().numpy()

            expert_names = ['defog', 'derain', 'desnow']
            ew = np.array(expert_weights)
            
            # 处理维度问题，确保是 [B, 3]
            if ew.ndim == 0:
                # 标量情况
                ew = ew.reshape(1, 1) 
            elif ew.ndim == 1:
                # 可能是 [3] (batch=1 squeeze后) 或 [B] (如果只有1个专家?)
                if ew.shape[0] == 3:
                     ew = ew.reshape(1, 3)
                else:
                     # 这里的逻辑取决于squeeze的行为，如果batch>1，squeeze不会去掉batch维
                     pass 
            
            # 保存专家权重到txt文件
            with open(weights_txt_path, "a") as f:
                for i in range(final_out.size(0)):
                    # 确保ew[i]有3个元素
                    if ew.ndim == 2 and ew.shape[1] == 3:
                        weights_info = ', '.join([f'{ew[i][j]:.4f}' for j in range(len(expert_names))])
                    else:
                        # fallback
                        weights_info = str(ew[i])
                        
                    f.write(f"{img_names[i]}, {weights_info}\n")

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
            
            print(f"Processed batch {idx+1}/{len(test_loader)}, PSNR: {batch_psnr:.2f}, SSIM: {batch_ssim:.4f}")

    avg_psnr = total_psnr / count if count > 0 else 0
    avg_ssim = total_ssim / count if count > 0 else 0
    print(f"[Test] Avg PSNR: {avg_psnr:.2f}, Avg SSIM: {avg_ssim:.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/MoEDataset', help='数据集路径')
    parser.add_argument('--weights_path', type=str, default='weather_removing_model/weights/moe_best_train2.pth', help='模型权重路径')
    parser.add_argument('--save_dir', type=str, default='weather_removing_model/results/predict', help='结果保存路径')
    parser.add_argument('--batch_size', type=int, default=1, help='批量大小')
    parser.add_argument('--num_workers', type=int, default=2, help='数据加载线程数')
    parser.add_argument('--class_num', type=int, default=3, help='天气类别数量')
    args = parser.parse_args()
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    main(args)
