import os, time, math
import numpy as np
import torch
import torch.nn.functional as F
from torch import optim, nn
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
import argparse
import random

from utils.SSIM import psnr, ssim
from models.experts.DEANetExpert import DEANetExpert
from utils.contrast_loss import ContrastLoss
from data.data_loader import singleDataset

# ----------------- 参数整合 -----------------
parser = argparse.ArgumentParser()
# --- 路径参数 ---
parser.add_argument('--data_dir', type=str, default='datasets/MoEDataset', help='数据集根目录')
parser.add_argument('--saved_log_dir', type=str, default='weather_removing/results/logs', help='保存结果的目录')
parser.add_argument('--saved_model_dir', type=str, default='weather_removing/weights/DEANet', help='保存模型的目录')

# --- 训练参数 ---
parser.add_argument('--device', type=str, default='Automatic detection')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--iters_per_epoch', type=int, default=5000)
parser.add_argument('--finer_eval_step', type=int, default=400000)
parser.add_argument('--bs', type=int, default=16, help='batch size')
parser.add_argument('--start_lr', default=4e-4, type=float, help='start learning rate')
parser.add_argument('--end_lr', default=1e-6, type=float, help='end learning rate')
parser.add_argument('--no_lr_sche', action='store_true', help='no lr cos schedule')
parser.add_argument('--use_warm_up', type=bool, default=False, help='using warm up in learning rate')
parser.add_argument('--w_loss_L1', default=1., type=float, help='weight of loss L1')
parser.add_argument('--w_loss_CR', default=0.1, type=float, help='weight of loss CR')
parser.add_argument('--resume', type=bool, default=False)
parser.add_argument('--pre_trained_model', type=str, default='null')
opt = parser.parse_args()
opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

# --------- 路径和目录创建 ---------
if not os.path.exists(opt.saved_log_dir):
    os.makedirs(opt.saved_log_dir)
if not os.path.exists(opt.saved_model_dir):
    os.makedirs(opt.saved_model_dir)


start_time = time.time()
steps = opt.iters_per_epoch * opt.epochs
T = steps


def lr_schedule_cosdecay(t, T, init_lr=opt.start_lr, end_lr=opt.end_lr):
    lr = end_lr + 0.5 * (init_lr - end_lr) * (1 + math.cos(t * math.pi / T))
    return lr


def train(net, loader_train, loader_val, optim, criterion):
    losses = []
    writer = SummaryWriter(log_dir=opt.tensorboard_log_dir)
    loss_log = {'L1': [], 'CR': [], 'total': []}
    loss_log_tmp = {'L1': [], 'CR': [], 'total': []}

    start_step = 0
    max_ssim = 0
    max_psnr = 0
    ssims = []
    psnrs = []

    loader_train_iter = iter(loader_train)

    for step in range(start_step + 1, steps + 1):
        net.train()
        lr = opt.start_lr
        if not opt.no_lr_sche:
            lr = lr_schedule_cosdecay(step, T)
            for param_group in optim.param_groups:
                param_group["lr"] = lr

        x, y = next(loader_train_iter)
        x = x.to(opt.device)
        y = y.to(opt.device)

        out = net(x)
        if opt.w_loss_L1 > 0:
            loss_L1 = criterion[0](out, y)
        if opt.w_loss_CR > 0:
            loss_CR = criterion[1](out, y, x)
        loss = opt.w_loss_L1 * loss_L1 + opt.w_loss_CR * loss_CR
        loss.backward()
        optim.step()
        optim.zero_grad()
        losses.append(loss.item())
        loss_log_tmp['L1'].append(loss_L1.item())
        loss_log_tmp['CR'].append(loss_CR.item())
        loss_log_tmp['total'].append(loss.item())

        # TensorBoard step-wise logs
        writer.add_scalar('train/loss_total', loss.item(), step)
        writer.add_scalar('train/loss_L1', loss_L1.item(), step)
        writer.add_scalar('train/loss_CR', loss_CR.item(), step)
        writer.add_scalar('train/lr', lr, step)

        if step % len(loader_train) == 0:
            loader_train_iter = iter(loader_train)
            for key in loss_log.keys():
                loss_log[key].append(np.average(np.array(loss_log_tmp[key])))
                loss_log_tmp[key] = []
            epoch_idx = int(step / len(loader_train))
            # TensorBoard epoch-wise averages
            writer.add_scalar('epoch/loss_total_avg', loss_log['total'][-1], epoch_idx)
            writer.add_scalar('epoch/loss_L1_avg', loss_log['L1'][-1], epoch_idx)
            writer.add_scalar('epoch/loss_CR_avg', loss_log['CR'][-1], epoch_idx)
        if (step % opt.iters_per_epoch == 0 and step <= opt.finer_eval_step) or (step > opt.finer_eval_step and (step - opt.finer_eval_step) % (5 * len(loader_train)) == 0):
            if step > opt.finer_eval_step:
                epoch = opt.finer_eval_step // opt.iters_per_epoch + (step - opt.finer_eval_step) // (5 * len(loader_train))
            else:
                epoch = int(step / opt.iters_per_epoch)
            with torch.no_grad():
                ssim_eval, psnr_eval = validate(net, loader_val)

            print(f'\nstep :{step} | epoch: {epoch} | ssim:{ssim_eval:.4f}| psnr:{psnr_eval:.4f}')
            # TensorBoard evaluation metrics
            writer.add_scalar('val/ssim', ssim_eval, epoch)
            writer.add_scalar('val/psnr', psnr_eval, epoch)

            if psnr_eval > max_psnr:
                max_ssim = max(max_ssim, ssim_eval)
                max_psnr = max(max_psnr, psnr_eval)
                print(
                    f'\n model saved at step :{step}| epoch: {epoch} | max_psnr:{max_psnr:.4f}| max_ssim:{max_ssim:.4f}')
                saved_best_model_path = os.path.join(opt.saved_model_dir, 'best.pk')
                torch.save({
                    'epoch': epoch,
                    'step': step,
                    'max_psnr': max_psnr,
                    'max_ssim': max_ssim,
                    'ssims': ssims,
                    'psnrs': psnrs,
                    'losses': losses,
                    'model': net.state_dict(),
                    'optimizer': optim.state_dict()
                }, saved_best_model_path)
            saved_single_model_path = os.path.join(opt.saved_model_dir, str(epoch) + '.pk')
            torch.save({
                'epoch': epoch,
                'step': step,
                'max_psnr': max_psnr,
                'max_ssim': max_ssim,
                'ssims': ssims,
                'psnrs': psnrs,
                'losses': losses,
                'model': net.state_dict(),
                'optimizer': optim.state_dict()
            }, saved_single_model_path)
            loader_train_iter = iter(loader_train)
    writer.close()

def pad_img(x, patch_size):
    _, _, h, w = x.size()
    mod_pad_h = (patch_size - h % patch_size) % patch_size
    mod_pad_w = (patch_size - w % patch_size) % patch_size
    x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
    return x

def validate(net, loader_val):
    net.eval()
    torch.cuda.empty_cache()
    ssims = []
    psnrs = []

    for i, (inputs, targets) in enumerate(loader_val):
        inputs = inputs.to(opt.device)
        targets = targets.to(opt.device)
        with torch.no_grad():
            H, W = inputs.shape[2:]
            inputs = pad_img(inputs, 4)
            pred = net(inputs).clamp(0, 1)
            pred = pred[:, :, :H, :W]
        ssim_tmp = ssim(pred, targets).item()
        psnr_tmp = psnr(pred, targets)
        ssims.append(ssim_tmp)
        psnrs.append(psnr_tmp)

    return np.mean(ssims), np.mean(psnrs)


def set_seed_torch(seed=2018):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    set_seed_torch(42)

    full_dataset = singleDataset(root_dir=opt.data_dir, weather_class='fog', crop_size=256)
    
    # 划分训练集和验证集
    validation_split = 0.2
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    loader_train = DataLoader(dataset=train_dataset, batch_size=opt.bs, shuffle=True, num_workers=4)
    loader_val = DataLoader(dataset=val_dataset, batch_size=1, shuffle=False, num_workers=4)

    # 在预训练阶段，不需要返回中间特征
    net = DEANetExpert(base_dim=32, return_features=False)
    net = net.to(opt.device)

    epoch_size = len(loader_train)
    print("epoch_size: ", epoch_size)
    if opt.device == 'cuda':
        net = torch.nn.DataParallel(net)
        cudnn.benchmark = True

    pytorch_total_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))

    criterion = []
    criterion.append(nn.L1Loss().to(opt.device))
    criterion.append(ContrastLoss(ablation=False))

    optimizer = optim.Adam(params=filter(lambda x: x.requires_grad, net.parameters()), lr=opt.start_lr, betas=(0.9, 0.999),
                           eps=1e-08)
    optimizer.zero_grad()
    train(net, loader_train, loader_val, optimizer, criterion)