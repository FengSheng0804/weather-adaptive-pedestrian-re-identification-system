import os
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision.transforms.functional import rotate, crop
from torchvision.transforms import ToTensor, RandomCrop

def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in ['.png', '.jpg', '.jpeg', '.bmp'])

class singleDataset(Dataset):
    def __init__(self, root_dir, weather_class, crop_size=256):
        # root_dir: MoEDataset根目录
        super(singleDataset, self).__init__()
        self.weather_path = os.path.join(root_dir, '1', weather_class)
        self.ground_truth_path = os.path.join(root_dir, '1', weather_class + '_ground_truth')
        self.weather_image_list = [f for f in os.listdir(self.weather_path) if is_image_file(f)]
        self.weather_image_list.sort()
        self.crop_size = crop_size

    def __getitem__(self, index):
        weather_image_name = self.weather_image_list[index]
        ground_truth_name = weather_image_name  # 名字相同
        
        weather_image_path = os.path.join(self.weather_path, weather_image_name)
        ground_truth_image_path = os.path.join(self.ground_truth_path, ground_truth_name)
        
        weather_image = Image.open(weather_image_path).convert('RGB')
        ground_truth_image = Image.open(ground_truth_image_path).convert('RGB')

        # 采用随机裁剪和旋转增强
        crop_params = RandomCrop.get_params(weather_image, [self.crop_size, self.crop_size])
        rotate_params = random.randint(0, 3) * 90
        # 随机裁剪
        weather_image = crop(weather_image, *crop_params)
        ground_truth_image = crop(ground_truth_image, *crop_params)
        # 随机旋转
        weather_image = rotate(weather_image, rotate_params)
        ground_truth_image = rotate(ground_truth_image, rotate_params)
        to_tensor = ToTensor()
        weather_image = to_tensor(weather_image)
        ground_truth_image = to_tensor(ground_truth_image)
        return weather_image, ground_truth_image

    def __len__(self):
        return len(self.weather_image_list)


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
                    # 每行可能有多个分数，全部处理成字典的形式
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) > 1:
                            scores.append({parts[i].split(':')[0]: float(parts[i].split(':')[1]) for i in range(len(parts))})
                        else:
                            scores.append({parts[0].split(':')[0]: float(parts[0].split(':')[1])})
            # 匹配图片
            img_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.png'))])
            for i, img_name in enumerate(img_files):
                input_path = os.path.join(input_dir, img_name)
                gt_path = os.path.join(gt_dir, img_name)
                score = scores[i] if i < len(scores) else None
                self.samples.append((input_path, gt_path, label, score))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        input_path, gt_path, label, score = self.samples[index]
        input_img = self.loader(input_path)
        gt_img = self.loader(gt_path)
        if self.transform:
            input_img = self.transform(input_img)
            gt_img = self.transform(gt_img)
        return input_img, gt_img, label, score