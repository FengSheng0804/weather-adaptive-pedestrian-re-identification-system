import os
import re
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms import ToTensor
from data import PairCompose, PairRandomCrop, PairToTensor

def default_loader(path):
    return Image.open(path).convert('RGB')

class MoETrainDataset(Dataset):
    """
    适配MoEDataset目录结构：
    datasets/MoEDataset/
        1/fog/xxx.jpg, scores.txt
        1/fog_ground_truth/xxx.jpg
        1/rain/xxx.jpg, scores.txt
        1/rain_ground_truth/xxx.jpg
        ...
        2/fog_rain/xxx.jpg, scores.txt
        2/fog_rain_ground_truth/xxx.jpg
        ...
    """
    def __init__(self, root_dir, crop_size=256, use_transform=True, loader=default_loader, scenario='all'):
        self.root_dir = root_dir
        # transform 设计为仿照 snow_removing_model，支持成对裁剪
        if use_transform:
            self.transform = PairCompose([
                PairRandomCrop(crop_size),
                PairToTensor()
            ])
        else:
            self.transform = None
        self.loader = loader
        self.samples = []  # (input_path, gt_path, label, score)

        # 支持多场景/all场景加载
        scenario_map = {
            'fog':      ('1/fog', '1/fog_ground_truth', '1/fog_mask', 'fog'),
            'rain':     ('1/rain', '1/rain_ground_truth', '1/rain_mask', 'rain'),
            'snow':     ('1/snow', '1/snow_ground_truth', '1/snow_mask', 'snow'),
            'fog_rain': ('2/fog_rain', '2/fog_rain_ground_truth', '2/fog_rain_mask', 'fog_rain'),
            'fog_snow': ('2/fog_snow', '2/fog_snow_ground_truth', '2/fog_snow_mask', 'fog_snow'),
            'rain_snow':('2/rain_snow', '2/rain_snow_ground_truth', '2/rain_snow_mask', 'rain_snow'),
            'fog_rain_snow': ('3/fog_rain_snow', '3/fog_rain_snow_ground_truth', '3/fog_rain_snow_mask', 'fog_rain_snow'),
        }
        if scenario == 'all':
            scenarios = list(scenario_map.values())
        else:
            scenarios = [scenario_map[scenario]]

        for input_rel, gt_rel, mask_rel, label in scenarios:
            input_dir = os.path.join(root_dir, 'train', input_rel)
            gt_dir = os.path.join(root_dir, 'train', gt_rel)
            mask_dir = os.path.join(root_dir, 'train', mask_rel)
            score_path = os.path.join(input_dir, 'scores.txt')
            # 读取分数
            scores = []
            if os.path.exists(score_path):
                with open(score_path, 'r') as f:
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            scores.append({parts[i].split(':')[0]: float(parts[i].split(':')[1]) for i in range(len(parts))})
                        else:
                            scores.append({parts[0].split(':')[0]: float(parts[0].split(':')[1])})
            # 统计所有可能的key（如 fog, rain, snow）
            all_keys = set(['fog', 'rain', 'snow'])
            for s in scores:
                all_keys.update(s.keys())
            # 补齐每个分数字典的key，缺失填0.0
            for s in scores:
                for k in all_keys:
                    if k not in s:
                        s[k] = 0.0
            # 匹配图片
            def natural_key(s):
                return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

            img_files = sorted(
                [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))],
                key=natural_key
            )
            
            for i, img_name in enumerate(img_files):
                input_path = os.path.join(input_dir, img_name)
                gt_path = os.path.join(gt_dir, img_name)
                # 若分数缺失，补一个全为0.0的字典
                if i < len(scores):
                    score = scores[i]
                else:
                    score = {k: 0.0 for k in all_keys}
                self.samples.append((input_path, gt_path, label, score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, gt_path, label, score = self.samples[index]
        input_img = self.loader(input_path)
        gt_img = self.loader(gt_path)
        if self.transform:
            input_img, gt_img = self.transform(input_img, gt_img)
        else:
            input_img = ToTensor()(input_img)
            gt_img = ToTensor()(gt_img)
        return input_img, gt_img, label, score
    
class MoETestDataset(Dataset):
    """
    适配MoEDataset目录结构：
    datasets/MoEDataset/
        test/weather_image/xxx.jpg, scores.txt
        test/ground_truth/xxx.jpg
    """
    def __init__(self, root_dir, crop_size=256, use_transform=False, loader=default_loader, scenario='all'):
        self.root_dir = root_dir
        # transform 设计为仿照 snow_removing_model，支持成对裁剪
        if use_transform:
            self.transform = PairCompose([
                PairRandomCrop(crop_size),
                PairToTensor()
            ])
        else:
            self.transform = None
        self.loader = loader
        self.samples = []  # (input_path, gt_path, img_name)

        input_dir = os.path.join(root_dir, "test", "weather_image")
        gt_dir = os.path.join(root_dir, "test", "ground_truth")
        score_path = os.path.join(input_dir, 'scores.txt')
        # 读取分数
        scores = []
        if os.path.exists(score_path):
            with open(score_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) > 1:
                        scores.append({parts[i].split(':')[0]: float(parts[i].split(':')[1]) for i in range(len(parts))})
                    else:
                        scores.append({parts[0].split(':')[0]: float(parts[0].split(':')[1])})
        # 统计所有可能的key（如 fog, rain, snow）
        all_keys = set(['fog', 'rain', 'snow'])
        for s in scores:
            all_keys.update(s.keys())
        # 补齐每个分数字典的key，缺失填0.0
        for s in scores:
            for k in all_keys:
                if k not in s:
                    s[k] = 0.0
        # 匹配图片
        img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
        for i, img_name in enumerate(img_files):
            input_path = os.path.join(input_dir, img_name)
            gt_path = os.path.join(gt_dir, img_name)
            # 推理阶段不返回分数（scores.txt 可能存在但不泄露）
            self.samples.append((input_path, gt_path, img_name))
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, index):
        input_path, gt_path, img_name = self.samples[index]
        input_img = self.loader(input_path)
        gt_img = self.loader(gt_path)
        if self.transform:
            input_img, gt_img = self.transform(input_img, gt_img)
        else:
            input_img = ToTensor()(input_img)
            gt_img = ToTensor()(gt_img)
        return input_img, gt_img, img_name