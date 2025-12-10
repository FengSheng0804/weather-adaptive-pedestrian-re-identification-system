import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from data.data_loader import MoEDataset
from models.moe import MoE
from .utils.SSIM import SSIM

parser = argparse.ArgumentParser(description="MoE Model Training")
parser.add_argument('--data_path', type=str, default='datasets/MoEDataset', help='训练数据路径')
parser.add_argument('--log_save_path', type=str, default='weather_removing_model/results/logs', help='日志保存路径')
parser.add_argument('--model_save_path', type=str, default='weather_removing_model/weights', help='模型保存路径')
parser.add_argument('--epochs', type=int, default=40, help='训练轮数')
parser.add_argument('--batch_size', type=int, default=8, help='批次大小')
parser.add_argument('--lr', type=float, default=1e-4, help='初始学习率')
parser.add_argument('--save_freq', type=int, default=1, help='模型保存频率')
parser.add_argument('--use_gpu', type=bool, default=True, help='是否使用GPU')
parser.add_argument('--local_rank', type=int, default=0, help='DDP本地rank')
parser.add_argument('--world_size', type=int, default=1, help='DDP总进程数')
parser.add_argument('--dist_backend', type=str, default='nccl', help='分布式后端')
parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:23456', help='分布式初始化URL')
args = parser.parse_args()


def train():
    # 分布式初始化
    if args.world_size > 1:
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url)
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
    else:
        device = torch.device('cuda' if args.use_gpu and torch.cuda.is_available() else 'cpu')

    # 日志与保存目录
    if args.local_rank == 0:
        os.makedirs(args.log_save_path, exist_ok=True)
        os.makedirs(args.model_save_path, exist_ok=True)
        writer = SummaryWriter(args.log_save_path)
    else:
        writer = None

    # 数据集与采样器
    train_dataset = MoEDataset(os.path.join(args.data_path, 'train'))
    val_dataset = MoEDataset(os.path.join(args.data_path, 'test'))
    if args.world_size > 1:
        train_sampler = DistributedSampler(train_dataset)
        val_sampler = DistributedSampler(val_dataset)
    else:
        train_sampler = None
        val_sampler = None
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, shuffle=(train_sampler is None), num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler, shuffle=False, num_workers=2, pin_memory=True)

    # 构建模型
    model = MoE().to(device)
    if args.world_size > 1:
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)
    # 损失函数
    criterion = SSIM().to(device)
    # 优化器与调度器
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    # 训练循环
    best_psnr = 0.0
    for epoch in range(args.epochs):
        if args.world_size > 1:
            train_sampler.set_epoch(epoch)
        model.train()
        for i, (input_img, gt_img) in enumerate(train_loader):
            input_img, gt_img = input_img.to(device), gt_img.to(device)
            optimizer.zero_grad()
            outputs = model(input_img)
            loss = -criterion(gt_img, outputs['final_output'])
            loss.backward()
            optimizer.step()
            # 训练日志
            if writer and i % 10 == 0:
                writer.add_scalar('train/loss', loss.item(), epoch * len(train_loader) + i)
        scheduler.step(loss.item())

        # 验证与保存
        model.eval()
        psnr_list = []
        with torch.no_grad():
            for input_img, gt_img in val_loader:
                input_img, gt_img = input_img.to(device), gt_img.to(device)
                outputs = model(input_img)
                pred = torch.clamp(outputs['final_output'], 0., 1.)
                mse = nn.functional.mse_loss(pred, gt_img)
                psnr = 10 * torch.log10(1.0 / mse)
                psnr_list.append(psnr.item())
            avg_psnr = sum(psnr_list) / len(psnr_list) if psnr_list else 0.0
            if writer:
                writer.add_scalar('val/psnr', avg_psnr, epoch)
        # 保存模型
        if args.local_rank == 0 and avg_psnr > best_psnr:
            best_psnr = avg_psnr
            torch.save(model.state_dict(), os.path.join(args.model_save_path, 'best_MoE.pth'))
        if args.local_rank == 0 and epoch % args.save_freq == 0:
            torch.save(model.state_dict(), os.path.join(args.model_save_path, f'net_epoch{epoch+1}.pth'))
        if writer:
            writer.flush()
    if writer:
        writer.close()
    if args.world_size > 1:
        dist.destroy_process_group()

if __name__ == "__main__":
    train()