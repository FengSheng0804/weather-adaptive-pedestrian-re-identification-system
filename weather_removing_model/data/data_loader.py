import os
from torch.utils.data import Dataset
from PIL import Image

def default_loader(path):
    return Image.open(path).convert('RGB')

class MoEDataset(Dataset):
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
    def __init__(self, root_dir, transform=None, loader=default_loader, scenario='all'):
        self.root_dir = root_dir
        self.transform = transform
        self.loader = loader
        self.samples = []  # (input_path, gt_path, label, score)

        # 支持多场景/all场景加载
        scenario_map = {
            'fog':      ('1/fog', '1/fog_ground_truth', 'fog'),
            'rain':     ('1/rain', '1/rain_ground_truth', 'rain'),
            'snow':     ('1/snow', '1/snow_ground_truth', 'snow'),
            'fog_rain': ('2/fog_rain', '2/fog_rain_ground_truth', 'fog_rain'),
            'fog_snow': ('2/fog_snow', '2/fog_snow_ground_truth', 'fog_snow'),
            'rain_snow':('2/rain_snow', '2/rain_snow_ground_truth', 'rain_snow'),
            'fog_rain_snow': ('3/fog_rain_snow', '3/fog_rain_snow_ground_truth', 'fog_rain_snow'),
        }
        if scenario == 'all':
            scenarios = list(scenario_map.values())
        else:
            scenarios = [scenario_map[scenario]]

        for input_rel, gt_rel, label in scenarios:
            input_dir = os.path.join(root_dir, input_rel)
            gt_dir = os.path.join(root_dir, gt_rel)
            score_path = os.path.join(input_dir, 'scores.txt')
            # 读取分数
            scores = []
            if os.path.exists(score_path):
                with open(score_path, 'r') as f:
                    scores = [line.strip() for line in f.readlines()]
            # 匹配图片
            img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
            for i, img_name in enumerate(img_files):
                input_path = os.path.join(input_dir, img_name)
                gt_path = os.path.join(gt_dir, img_name)
                score = scores[i] if i < len(scores) else None
                self.samples.append((input_path, gt_path, label, score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        input_path, gt_path, label, score = self.samples[idx]
        input_img = self.loader(input_path)
        gt_img = self.loader(gt_path)
        if self.transform:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
        return input_img, gt_img, label, score