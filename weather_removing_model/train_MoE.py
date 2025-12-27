import os
import argparse
from collections import OrderedDict

from matplotlib import pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

from models.moe import MoE
from data.data_loader import MoETrainDataset
from utils.SSIM import ssim, psnr
from utils.contrast_loss import ContrastLoss
import time

# 去除module前缀，使DEANet可以加载
def strip_module_prefix(state_dict):
    """Remove a leading 'module.' in state_dict keys if present."""
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

# 加载预训练权重，支持key提取和module前缀去除
def load_pretrained(expert, ckpt_path, key=None):
    """Load pretrained weights with optional key and module prefix stripping."""
    if not os.path.exists(ckpt_path):
        print(f"[WARN] checkpoint not found: {ckpt_path}")
        return
    ckpt = torch.load(ckpt_path, map_location='cpu')
    if key is not None:
        ckpt = ckpt[key]
    ckpt = strip_module_prefix(ckpt)
    missing, unexpected = expert.load_state_dict(ckpt, strict=False)
    if missing:
        print(f"[INFO] missing keys ({len(missing)}): {missing[:8]}{' ...' if len(missing)>8 else ''}")
    if unexpected:
        print(f"[INFO] unexpected keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected)>8 else ''}")

# 构建数据加载器
def build_dataloaders(data_root, batch_size, num_workers, train_scenario='all'):
    train_set = MoETrainDataset(root_dir=data_root, crop_size=256, use_transform=True, scenario=train_scenario)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    return train_loader


def train_worker(local_rank, args):
    # device setup
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device('cuda', local_rank)
    else:
        device = torch.device('cpu')

    # Model
    model = MoE(args.class_num)
    # Load expert pretrained weights
    load_pretrained(model.derain_expert, args.prenet_ckpt, key=None)  # PReNet: plain state_dict
    load_pretrained(model.defog_expert, args.deanet_ckpt, key='model')  # DEANet: saved with {'model': ...}
    load_pretrained(model.desnow_expert, args.convir_ckpt, key='model')  # ConvIR: saved with {'model': ...}
    model.to(device)

    # Parallelize
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None,
            output_device=local_rank if device.type == 'cuda' else None,
            find_unused_parameters=True,
        )
    else:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)

    # Param groups: separate experts vs gate/fusion/reconstruction
    main_params = []  # gate, fusion, reconstruction, residual
    expert_pretrained_params = []  # derain/defog/desnow pretrained parts
    expert_newlayer_params = []    # newly added layers (e.g., feature_extractor, moe_adapter)
    # 对于多卡，需要访问 module 属性
    m = model.module if hasattr(model, 'module') else model
    main_params += list(m.moe_gate.parameters())
    main_params += list(m.feature_fusion.parameters())
    main_params += list(m.reconstruction_net.parameters())
    main_params += list(m.residual_conv.parameters())

    def split_expert_params(expert, new_prefixes=("feature_extractor", "moe_adapter")):
        pretrained, newlayers = [], []
        for name, p in expert.named_parameters():
            if any(name.startswith(pref) or ("."+pref+".") in name for pref in new_prefixes):
                newlayers.append(p)
            else:
                pretrained.append(p)
        return pretrained, newlayers

    p_derain_pre, p_derain_new = split_expert_params(m.derain_expert)
    p_defog_pre,  p_defog_new  = split_expert_params(m.defog_expert)
    p_desnow_pre, p_desnow_new = split_expert_params(m.desnow_expert)
    expert_pretrained_params += p_derain_pre + p_defog_pre + p_desnow_pre
    expert_newlayer_params   += p_derain_new + p_defog_new + p_desnow_new

    # Freeze only experts' pretrained parts initially; keep new layers trainable
    if getattr(args, 'freeze_experts_epochs', 0) > 0:
        for p in expert_pretrained_params:
            p.requires_grad = False
        for p in expert_newlayer_params:
            p.requires_grad = True

    # Build optimizer with param groups
    optimizer = torch.optim.Adam([
        { 'params': main_params, 'lr': args.main_lr, 'betas': (0.9, 0.999) },
        { 'params': expert_newlayer_params, 'lr': args.experts_lr, 'betas': (0.9, 0.999) },
        { 'params': expert_pretrained_params, 'lr': args.experts_lr, 'betas': (0.9, 0.999) }
    ])
    
    # Loss functions
    l1_loss = torch.nn.L1Loss()
    contrast_loss = ContrastLoss()

    # Data
    if args.distributed:
        # use DistributedSampler for DDP
        train_set = MoETrainDataset(root_dir=args.data_dir, crop_size=256, use_transform=True, scenario=args.scenario)
        train_sampler = DistributedSampler(train_set, shuffle=True)
        train_loader = DataLoader(train_set, batch_size=args.batch_size, sampler=train_sampler,
                                  num_workers=args.num_workers, pin_memory=True, drop_last=True)
    else:
        train_loader = build_dataloaders(args.data_dir, args.batch_size, args.num_workers, args.scenario)

    global_step = 0
    best_psnr = 0.0

    os.makedirs(args.save_dir, exist_ok=True)

    # master-only logging
    is_master = (not args.distributed) or (dist.is_initialized() and dist.get_rank() == 0)

    # TensorBoard writer (master rank only)
    writer = None
    if is_master:
        log_dir = args.log_dir if hasattr(args, 'log_dir') and args.log_dir else os.path.join(args.save_dir, 'runs')
        os.makedirs(log_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=log_dir)

    model.train()
    for epoch in range(1, args.epochs + 1):
        # Unfreeze experts at configured epoch
        if getattr(args, 'freeze_experts_epochs', 0) > 0 and epoch == args.freeze_experts_epochs + 1:
            # unfreeze pretrained expert parts
            for p in expert_pretrained_params:
                p.requires_grad = True
            # ensure optimizer has correct learning rates (in case modified)
            if len(optimizer.param_groups) >= 3:
                optimizer.param_groups[0]['lr'] = args.main_lr
                optimizer.param_groups[1]['lr'] = args.experts_lr  # new layers
                optimizer.param_groups[2]['lr'] = args.experts_lr  # pretrained parts
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        for batch in train_loader:
            inputs, targets, _, scores = batch

            # # 显示输入的batch个图像在一张图像上，并把图像对应的scores标在图像的左上角
            # import matplotlib.pyplot as plt
            # import torchvision
            # import numpy as np
            # if is_master:
            #     nrow = 4
            #     batch_imgs = inputs.cpu()
            #     # 处理分数
            #     if isinstance(scores, dict):
            #         score_keys = ['fog', 'rain', 'snow']
            #         batch_scores = torch.stack([scores[k].cpu() for k in score_keys], dim=1).numpy()  # [B,3]
            #     elif scores is None:
            #         batch_scores = None
            #     else:
            #         batch_scores = np.array(scores)
            #         if batch_scores.ndim == 1:
            #             batch_scores = batch_scores.reshape(-1, 3)
            #     # 绘制每张图像并标注分数
            #     imgs = []
            #     for i in range(batch_imgs.size(0)):
            #         img = batch_imgs[i]
            #         img = torchvision.transforms.functional.to_pil_image(img)
            #         img = np.array(img)
            #         fig, ax = plt.subplots()
            #         ax.imshow(img)
            #         ax.axis('off')
            #         # 标注分数
            #         if batch_scores is not None:
            #             score_str = ', '.join([f'{k}:{batch_scores[i, j]:.2f}' for j, k in enumerate(['fog', 'rain', 'snow'])])
            #             ax.text(2, 12, score_str, color='yellow', fontsize=10, bbox=dict(facecolor='black', alpha=0.5, pad=1))
            #         fig.canvas.draw()
            #         img_ann = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            #         img_ann = img_ann.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            #         imgs.append(img_ann)
            #         plt.close(fig)
            #     # 拼接成网格
            #     rows = []
            #     for i in range(0, len(imgs), nrow):
            #         row_imgs = imgs[i:i+nrow]
            #         # 补齐空白
            #         while len(row_imgs) < nrow:
            #             row_imgs.append(np.zeros_like(row_imgs[0]))
            #         rows.append(np.concatenate(row_imgs, axis=1))
            #     grid_img = np.concatenate(rows, axis=0)
            #     plt.figure(figsize=(8, 8))
            #     plt.imshow(grid_img)
            #     plt.axis('off')
            #     plt.title(f'Input Batch at Step {global_step}')
            #     plt.show()

            inputs = inputs.to(device)
            targets = targets.to(device)
            # 支持score为dict of tensor（如{'fog':tensor([..]),...}），或list of dict
            if isinstance(scores, dict):
                score_keys = ['fog', 'rain', 'snow']
                score_tensor = torch.stack([scores[k].to(inputs.device).to(inputs.dtype) for k in score_keys], dim=1)  # [B,3]
            elif scores is None:
                score_tensor = None
            else:
                score_tensor = torch.tensor(scores, dtype=inputs.dtype, device=inputs.device).view(-1, 3)

            # mask softmax归一化，保证0分项权重为0，其余归一化
            if score_tensor is not None:
                mask = (score_tensor > 0).float()
                masked_score = score_tensor * mask + (1 - mask) * -1e9
                score_tensor = F.softmax(masked_score, dim=1) * mask
                score_sum = score_tensor.sum(dim=1, keepdim=True) + 1e-8
                score_tensor = score_tensor / score_sum

            optimizer.zero_grad()
            outputs = model(inputs, score=score_tensor)
            final_out = outputs['final_output']
            features = outputs['fused_features']
            gate_weights = outputs['expert_weights']  # [B, num_experts] or [B, 3, 1, 1, 1]
            feature_weights = outputs.get('feature_weights', None)  # [B, num_experts]，新版moe.py已返回
            # === 计算各专家loss ===
            defog_out = outputs.get('defog_output', None)
            derain_out = outputs.get('derain_output', None)
            desnow_out = outputs.get('desnow_output', None)
            # L1
            defog_l1 = l1_loss(defog_out, targets) if defog_out is not None else torch.tensor(0.0, device=inputs.device)
            derain_l1 = l1_loss(derain_out, targets) if derain_out is not None else torch.tensor(0.0, device=inputs.device)
            desnow_l1 = l1_loss(desnow_out, targets) if desnow_out is not None else torch.tensor(0.0, device=inputs.device)
            # SSIM
            defog_ssim = 1 - ssim(defog_out, targets) if defog_out is not None else torch.tensor(0.0, device=inputs.device)
            derain_ssim = 1 - ssim(derain_out, targets) if derain_out is not None else torch.tensor(0.0, device=inputs.device)
            desnow_ssim = 1 - ssim(desnow_out, targets) if desnow_out is not None else torch.tensor(0.0, device=inputs.device)
            # Contrast
            defog_contrast = contrast_loss(defog_out, targets, inputs) if defog_out is not None else torch.tensor(0.0, device=inputs.device)
            derain_contrast = contrast_loss(derain_out, targets, inputs) if derain_out is not None else torch.tensor(0.0, device=inputs.device)
            desnow_contrast = contrast_loss(desnow_out, targets, inputs) if desnow_out is not None else torch.tensor(0.0, device=inputs.device)
            
            # 可视化gate输出与score的关系
            if is_master and writer is not None and global_step % args.log_interval == 0:
                # 展开gate_weights为[B, num_experts]
                gw = gate_weights
                if gw.dim() > 2:
                    gw = gw.view(gw.size(0), -1)
                # 只支持单一score（如强度），多score可扩展
                if score_tensor is not None and score_tensor.size(1) == 3:
                    for i in range(gw.size(1)):
                        for j, k in enumerate(['fog', 'rain', 'snow']):
                            writer.add_scalars(f'gate_vs_score/scatter_expert{i+1}_vs_{k}', {f'score_{kk}': gw[kk, i].item() for kk in range(gw.size(0))}, global_step)

            # calculate unified loss (score加权)
            if score_tensor is not None:
                norm_score = (score_tensor - score_tensor.min(dim=0, keepdim=True)[0]) / (score_tensor.max(dim=0, keepdim=True)[0] - score_tensor.min(dim=0, keepdim=True)[0] + 1e-8)
                sample_weights = 1.0 + norm_score.max(dim=1)[0]
            else:
                sample_weights = torch.ones(inputs.size(0), device=inputs.device)

            # calculate losses
            l1 = l1_loss(final_out, targets)
            ssim_val = 1 - ssim(final_out, targets)
            # 对比损失，使用输出、GT、输入
            contrast = contrast_loss(final_out, targets, inputs)
            base_loss = (0.8 * l1 * sample_weights).mean() + 0.15 * ssim_val + 0.05 * contrast

            # 负载均衡：对门控权重添加正则，避免单专家主导
            # 1) 每样本熵正则，鼓励分布更均匀
            gate_flat = gate_weights
            if gate_flat.dim() > 2:
                gate_flat = gate_flat.view(gate_flat.size(0), -1)  # [B, num_experts]
            eps = 1e-8
            gate_entropy = - (gate_flat * (gate_flat + eps).log()).sum(dim=1).mean()

            # 2) 批次均值与均匀分布的KL，鼓励整体负载均衡
            avg_w = gate_flat.mean(dim=0)  # [num_experts]
            uniform = torch.full_like(avg_w, 1.0 / avg_w.numel())
            kl_uniform = (avg_w * (avg_w + eps).log() - avg_w * (uniform + eps).log()).sum()

            # 门控与score对齐损失（KL散度）
            align_loss = 0.0
            # 这里使用到的score_tensor是归一化后的
            if score_tensor is not None:
                # KL(gate_weights || score_tensor)
                # 归一化，避免0导致log(0)
                gate_prob = gate_flat / (gate_flat.sum(dim=1, keepdim=True) + eps)
                score_prob = score_tensor / (score_tensor.sum(dim=1, keepdim=True) + eps)
                # 避免log(0)，加eps
                kl_align = (gate_prob * (gate_prob.add(eps).log() - score_prob.add(eps).log())).sum(dim=1).mean()
                align_loss = kl_align
                # 可选：feature_weights 也加对齐损失
                if feature_weights is not None:
                    feat_flat = feature_weights
                    if feat_flat.dim() > 2:
                        feat_flat = feat_flat.view(feat_flat.size(0), -1)
                    feat_prob = feat_flat / (feat_flat.sum(dim=1, keepdim=True) + eps)
                    kl_feat = (feat_prob * (feat_prob.add(eps).log() - score_prob.add(eps).log())).sum(dim=1).mean()
                    align_loss = align_loss + kl_feat

            # 合成负载均衡正则+对齐损失
            lb_loss = - args.gate_entropy_coef * gate_entropy + args.gate_balance_coef * kl_uniform
            align_coef = getattr(args, 'gate_align_coef', 1e-2)  # 可通过命令行调整
            loss = base_loss + lb_loss + align_coef * align_loss
            
            # backpropagation
            loss.backward()
            # update parameters
            optimizer.step()

            if is_master and global_step % args.log_interval == 0:
                with torch.no_grad():
                    cur_psnr = psnr(final_out.clamp(0, 1), targets)
                    cur_ssim = ssim(final_out.clamp(0, 1), targets).item()
                # Console log
                if 'last_time' not in locals():
                    last_time = time.time()
                cur_time = time.time()
                elapsed = cur_time - last_time
                print(f"Epoch {epoch} Step {global_step} | loss {loss.item():.4f} | PSNR {cur_psnr:.2f} | SSIM {cur_ssim:.4f} | time {elapsed:.2f}s")
                print(f"  [Expert L1] defog: {defog_l1.item():.4f}, derain: {derain_l1.item():.4f}, desnow: {desnow_l1.item():.4f}")
                print(f"  [Expert SSIM] defog: {defog_ssim.item():.4f}, derain: {derain_ssim.item():.4f}, desnow: {desnow_ssim.item():.4f}")
                print(f"  [Expert Contrast] defog: {defog_contrast.item():.4f}, derain: {derain_contrast.item():.4f}, desnow: {desnow_contrast.item():.4f}")
                last_time = cur_time
                # TensorBoard scalars
                if writer is not None:
                    writer.add_scalar('train/loss', loss.item(), global_step)
                    writer.add_scalar('train/psnr', cur_psnr, global_step)
                    writer.add_scalar('train/ssim', cur_ssim, global_step)
                    # Learning rate (handle schedulers later if added)
                    writer.add_scalar('train/lr_main', optimizer.param_groups[0]['lr'], global_step)
                    if len(optimizer.param_groups) > 1:
                        writer.add_scalar('train/lr_experts', optimizer.param_groups[1]['lr'], global_step)

            if is_master and global_step % 200 == 0 and score_tensor is not None:
                gw = gate_weights
                if gw.dim() > 2:
                    gw = gw.view(gw.size(0), -1)
                st = score_tensor
                gw_np = gw.detach().cpu().numpy()
                st_np = st.detach().cpu().numpy()
                for i, name in enumerate(['fog', 'rain', 'snow']):
                    plt.figure()
                    plt.scatter(st_np[:, i], gw_np[:, i], alpha=0.5)
                    plt.xlabel(f'{name} score')
                    plt.ylabel(f'gate weight {name}')
                    plt.title(f'Scatter: {name} score vs gate weight')
                    plt.savefig(f'gate_vs_score_{name}_{global_step}.png')
                    plt.close()
                # 计算相关系数
                corr = np.corrcoef(st_np.T, gw_np.T)[:3, 3:]
                print(f'[Corr] step {global_step}:\n', corr)


            # Image visualization at lower frequency
            if is_master and writer is not None and hasattr(args, 'vis_interval') and global_step % args.vis_interval == 0:
                with torch.no_grad():
                    n = min(getattr(args, 'vis_count', 4), inputs.size(0))
                    inp_vis = inputs[:n].detach().clamp(0, 1)
                    out_vis = final_out[:n].detach().clamp(0, 1)
                    tgt_vis = targets[:n].detach().clamp(0, 1)
                    # 取专家与MoE中间输出（可能不存在则跳过）
                    moe_vis = outputs.get('moe_output', None)
                    defog_vis = outputs.get('defog_output', None)
                    derain_vis = outputs.get('derain_output', None)
                    desnow_vis = outputs.get('desnow_output', None)
                    if moe_vis is not None:
                        moe_vis = moe_vis[:n].detach().clamp(0, 1)
                    if defog_vis is not None:
                        defog_vis = defog_vis[:n].detach().clamp(0, 1)
                    if derain_vis is not None:
                        derain_vis = derain_vis[:n].detach().clamp(0, 1)
                    if desnow_vis is not None:
                        desnow_vis = desnow_vis[:n].detach().clamp(0, 1)
                    writer.add_images('train/input', inp_vis, global_step)
                    writer.add_images('train/output', out_vis, global_step)
                    writer.add_images('train/target', tgt_vis, global_step)
                    if moe_vis is not None:
                        writer.add_images('train/moe_output', moe_vis, global_step)
                    if defog_vis is not None:
                        writer.add_images('train/expert_defog', defog_vis, global_step)
                    if derain_vis is not None:
                        writer.add_images('train/expert_derain', derain_vis, global_step)
                    if desnow_vis is not None:
                        writer.add_images('train/expert_desnow', desnow_vis, global_step)

            global_step += 1

        # simple eval on training subset (fast metric)
        with torch.no_grad():
            sample_inputs, sample_targets, _, sample_scores = next(iter(train_loader))
            sample_inputs = sample_inputs.to(device)
            sample_targets = sample_targets.to(device)
            if isinstance(sample_scores, dict):
                score_keys = ['fog', 'rain', 'snow']
                sample_score_tensor = torch.stack([sample_scores[k].to(sample_inputs.device).to(sample_inputs.dtype) for k in score_keys], dim=1)
            elif isinstance(sample_scores, list) and isinstance(sample_scores[0], dict):
                score_keys = ['fog', 'rain', 'snow']
                sample_score_tensor = torch.tensor([[float(s.get(k, 0.0)) for k in score_keys] for s in sample_scores], dtype=sample_inputs.dtype, device=sample_inputs.device)
            elif sample_scores is None:
                sample_score_tensor = None
            else:
                sample_score_tensor = torch.tensor(sample_scores, dtype=sample_inputs.dtype, device=sample_inputs.device).view(-1, 3)
            eval_out = model(sample_inputs, score=sample_score_tensor)['final_output'].clamp(0, 1)
            eval_psnr = psnr(eval_out, sample_targets)
        # Eval visualizations
        if is_master and writer is not None:
            n_eval = min(getattr(args, 'vis_count', 5), sample_inputs.size(0))
            writer.add_images('eval/input', sample_inputs[:n_eval].detach().clamp(0, 1), epoch)
            writer.add_images('eval/output', eval_out[:n_eval].detach().clamp(0, 1), epoch)
            writer.add_images('eval/target', sample_targets[:n_eval].detach().clamp(0, 1), epoch)
        if is_master and eval_psnr > best_psnr:
            best_psnr = eval_psnr
            save_path = os.path.join(args.save_dir, 'moe_best.pth')
            to_save = (model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
            torch.save({'model': to_save, 'psnr': best_psnr, 'epoch': epoch}, save_path)
            print(f"[SAVE] epoch {epoch} psnr {best_psnr:.2f} -> {save_path}")
        # Log epoch-level eval metric
        if is_master and writer is not None:
            writer.add_scalar('eval/psnr', eval_psnr, epoch)

        # optional periodic save
        if is_master and epoch % args.save_interval == 0:
            save_path = os.path.join(args.save_dir, f'moe_epoch_{epoch}.pth')
            to_save = (model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
            torch.save({'model': to_save, 'epoch': epoch}, save_path)

    # final save
    if is_master:
        final_path = os.path.join(args.save_dir, 'moe_final.pth')
        to_save = (model.module.state_dict() if hasattr(model, 'module') else model.state_dict())
        torch.save({'model': to_save, 'epoch': args.epochs}, final_path)
        print(f"[DONE] saved final model to {final_path}")
        if writer is not None:
            writer.close()


def init_distributed(args):
    if args.distributed and not dist.is_initialized():
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=int(os.environ.get('RANK', '0')))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='datasets/MoEDataset', help='MoE dataset root')
    parser.add_argument('--scenario', type=str, default='all', help='scenario key or all')
    parser.add_argument('--class_num', type=int, default=3, help='number of weather types/classes')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--epochs', type=int, default=20)
    # Learning rates
    parser.add_argument('--main_lr', type=float, default=1e-4, help='LR for gate/fusion/reconstruction')
    parser.add_argument('--experts_lr', type=float, default=1e-5, help='LR for experts during finetune')
    # Gate Load Balancing Regularization
    parser.add_argument('--gate_entropy_coef', type=float, default=5e-2, help='coef for gate entropy regularization')
    parser.add_argument('--gate_balance_coef', type=float, default=5e-2, help='coef for batch-level uniform KL')
    # Freeze-then-unfreeze config
    parser.add_argument('--freeze_experts_epochs', type=int, default=5, help='Freeze experts for N initial epochs')
    parser.add_argument('--log_interval', type=int, default=50)
    parser.add_argument('--vis_interval', type=int, default=200, help='steps between image visualizations')
    parser.add_argument('--vis_count', type=int, default=5, help='number of images to visualize per write')
    parser.add_argument('--save_interval', type=int, default=5, help='epochs between saving model checkpoints')
    parser.add_argument('--save_dir', type=str, default='weather_removing_model/weights')
    parser.add_argument('--log_dir', type=str, default='weather_removing_model/results/logs')
    parser.add_argument('--prenet_ckpt', type=str, default='weather_removing_model/weights/PReNet_pretrained.pth')
    parser.add_argument('--deanet_ckpt', type=str, default='weather_removing_model/weights/DEANet_pretrained.pk')
    parser.add_argument('--convir_ckpt', type=str, default='weather_removing_model/weights/ConvIR_pretrained.pkl')
    # DistrubutedDataParallel paramter config
    parser.add_argument('--distributed', action='store_true', help='Use DistributedDataParallel')
    parser.add_argument('--world_size', type=int, default=1, help='Number of processes for DDP')
    parser.add_argument('--dist_backend', type=str, default='nccl', help='DDP backend (nccl for CUDA)')
    parser.add_argument('--dist_url', type=str, default='tcp://127.0.0.1:29500', help='Init method URL for DDP')
    # 新增门控对齐损失系数参数
    parser.add_argument('--gate_align_coef', type=float, default=1e-2, help='coef for gate-score alignment loss (KL)')
    args = parser.parse_args()

    if args.distributed:
        n_gpus = torch.cuda.device_count()
        if args.world_size == 1:
            args.world_size = max(1, n_gpus)
        os.environ['WORLD_SIZE'] = str(args.world_size)
        # recommend torchrun externally; fallback to local spawn
        init_distributed(args)
        local_rank = int(os.environ.get('LOCAL_RANK', '0'))
        train_worker(local_rank, args)
        if dist.is_initialized():
            dist.destroy_process_group()
    else:
        local_rank = 0
        train_worker(local_rank, args)


if __name__ == '__main__':
    main()