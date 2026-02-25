import os
import sys
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from torchvision.utils import save_image
import cv2
import time

# Add the project root to sys.path to import from weather_removing_model
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from weather_removing_model.models.moe import MoE
from weather_removing_model.utils.SSIM import ssim, psnr

def strip_module_prefix(state_dict):
    from collections import OrderedDict
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        if k.startswith('module.'):
            new_sd[k[7:]] = v
        else:
            new_sd[k] = v
    return new_sd

# Global model instance to avoid reloading
_moe_model = None
_device = None

def get_moe_model():
    global _moe_model, _device
    if _moe_model is None:
        _device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        _moe_model = MoE(score_dim=3)
        
        if hasattr(_moe_model, 'reconstruction_net') and _moe_model.reconstruction_net[0].in_channels != 67:
            _moe_model.reconstruction_net[0] = torch.nn.Conv2d(67, 64, 3, padding=1)
            
        weights_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../weather_removing_model/weights/moe_best.pth'))
        
        # Fallback to other weights if moe_best.pth doesn't exist
        if not os.path.exists(weights_path):
            weights_dir = os.path.dirname(weights_path)
            if os.path.exists(weights_dir):
                pth_files = [f for f in os.listdir(weights_dir) if f.endswith('.pth')]
                if pth_files:
                    weights_path = os.path.join(weights_dir, pth_files[0])
        
        if os.path.exists(weights_path):
            checkpoint = torch.load(weights_path, map_location='cpu')
            state_dict = checkpoint['model'] if 'model' in checkpoint else checkpoint
            state_dict = strip_module_prefix(state_dict)
            _moe_model.load_state_dict(state_dict, strict=False)
        else:
            print(f"Warning: MoE weights not found at {weights_path}")
            
        _moe_model.to(_device)
        _moe_model.eval()
    return _moe_model, _device

def process_image(input_path, output_path):
    model, device = get_moe_model()
    
    # Load image
    img = Image.open(input_path).convert('RGB')
    transform = transforms.ToTensor()
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    # Pad to multiple of 32
    factor = 32
    h, w = input_tensor.shape[2], input_tensor.shape[3]
    padh = (factor - h % factor) % factor
    padw = (factor - w % factor) % factor
    
    if padh > 0 or padw > 0:
        input_tensor = torch.nn.functional.pad(input_tensor, (0, padw, 0, padh), 'reflect')
        
    with torch.no_grad():
        outputs = model(input_tensor)
        final_out = outputs['final_output'][:, :, :h, :w].clamp(0, 1)
        
        # Expert outputs
        defog_out = outputs['defog_output'][:, :, :h, :w].clamp(0, 1)
        derain_out = outputs['derain_output'][:, :, :h, :w].clamp(0, 1)
        desnow_out = outputs['desnow_output'][:, :, :h, :w].clamp(0, 1)
        
    # Save output
    save_image(final_out[0], output_path)
    
    # Save expert outputs
    base_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    experts_dir = os.path.join(base_dir, 'experts')
    os.makedirs(experts_dir, exist_ok=True)
    
    save_image(defog_out[0], os.path.join(experts_dir, f'defog_{filename}'))
    save_image(derain_out[0], os.path.join(experts_dir, f'derain_{filename}'))
    save_image(desnow_out[0], os.path.join(experts_dir, f'desnow_{filename}'))
    
    # Try to find ground truth image for accurate PSNR/SSIM calculation
    original_filename = os.path.basename(input_path)
    if '_' in original_filename:
        # Remove the uuid prefix
        original_filename = original_filename.split('_', 1)[1]
        
    gt_path = os.path.abspath(os.path.join(os.path.dirname(__file__), f'../../datasets/MoEDataset/test/ground_truth/{original_filename}'))
    
    if os.path.exists(gt_path):
        gt_img = Image.open(gt_path).convert('RGB')
        gt_tensor = transform(gt_img).unsqueeze(0).to(device)
        # Ensure gt_tensor has the same shape as final_out
        gt_cropped = gt_tensor[:, :, :h, :w]
        psnr_val = psnr(final_out, gt_cropped)
        ssim_val = ssim(final_out, gt_cropped).item()
    else:
        # Fallback: calculate against input image, but this will be low
        input_cropped = input_tensor[:, :, :h, :w]
        psnr_val = psnr(final_out, input_cropped)
        ssim_val = ssim(final_out, input_cropped).item()
    
    return psnr_val, ssim_val

def process_video(input_path, output_path, progress_callback=None):
    model, device = get_moe_model()
    
    cap = cv2.VideoCapture(input_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Expert writers
    base_dir = os.path.dirname(output_path)
    filename = os.path.basename(output_path)
    experts_dir = os.path.join(base_dir, 'experts')
    os.makedirs(experts_dir, exist_ok=True)
    
    out_defog = cv2.VideoWriter(os.path.join(experts_dir, f'defog_{filename}'), fourcc, fps, (width, height))
    out_derain = cv2.VideoWriter(os.path.join(experts_dir, f'derain_{filename}'), fourcc, fps, (width, height))
    out_desnow = cv2.VideoWriter(os.path.join(experts_dir, f'desnow_{filename}'), fourcc, fps, (width, height))
    
    transform = transforms.ToTensor()
    
    frame_count = 0
    total_psnr = 0
    total_ssim = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        input_tensor = transform(img).unsqueeze(0).to(device)
        
        # Pad
        factor = 32
        h, w = input_tensor.shape[2], input_tensor.shape[3]
        padh = (factor - h % factor) % factor
        padw = (factor - w % factor) % factor
        
        if padh > 0 or padw > 0:
            input_tensor = torch.nn.functional.pad(input_tensor, (0, padw, 0, padh), 'reflect')
            
        with torch.no_grad():
            outputs = model(input_tensor)
            final_out = outputs['final_output'][:, :, :h, :w].clamp(0, 1)
            
            defog_out = outputs['defog_output'][:, :, :h, :w].clamp(0, 1)
            derain_out = outputs['derain_output'][:, :, :h, :w].clamp(0, 1)
            desnow_out = outputs['desnow_output'][:, :, :h, :w].clamp(0, 1)
            
        # Calculate metrics
        input_cropped = input_tensor[:, :, :h, :w]
        total_psnr += psnr(final_out, input_cropped)
        total_ssim += ssim(final_out, input_cropped).item()
            
        # Convert back to BGR for OpenCV
        def to_bgr(tensor):
            t = tensor[0].cpu().numpy()
            t = np.transpose(t, (1, 2, 0)) * 255.0
            return cv2.cvtColor(t.astype(np.uint8), cv2.COLOR_RGB2BGR)

        out.write(to_bgr(final_out))
        out_defog.write(to_bgr(defog_out))
        out_derain.write(to_bgr(derain_out))
        out_desnow.write(to_bgr(desnow_out))
        
        frame_count += 1
        
        if progress_callback and total_frames > 0:
            should_continue = progress_callback(frame_count, total_frames)
            if should_continue is False:
                cap.release()
                out.release()
                out_defog.release()
                out_derain.release()
                out_desnow.release()
                return None
            
    cap.release()
    out.release()
    out_defog.release()
    out_derain.release()
    out_desnow.release()
    
    avg_psnr = total_psnr / frame_count if frame_count > 0 else 0
    avg_ssim = total_ssim / frame_count if frame_count > 0 else 0
    
    return avg_psnr, avg_ssim

def run_moe_inference(input_path, output_path, progress_callback=None):
    ext = input_path.rsplit('.', 1)[1].lower()
    if ext in ['mp4', 'avi', 'mov']:
        return process_video(input_path, output_path, progress_callback)
    else:
        return process_image(input_path, output_path)
