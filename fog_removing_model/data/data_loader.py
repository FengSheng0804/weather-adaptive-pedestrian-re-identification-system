import os, random
from torch.utils.data import Subset
import torch.utils.data as data
from PIL import Image
from torchvision.transforms.functional import hflip, rotate, crop
from torchvision.transforms import ToTensor, RandomCrop, Resize

IMG_EXTENSIONS = ['.png', '.jpg', '.jpeg', '.bmp']


def is_image_file(filename):
    return any(filename.lower().endswith(ext) for ext in IMG_EXTENSIONS)


def get_clear_name(hazy_name):
    # 取下划线前缀作为clear图名，后缀为.jpg
    # 已知foggy_image的名字是fog_train_1_1.jpg的形式
    # ground_truth的名字是fog_train_1.jpg的形式
    return '_'.join(hazy_name.split('_')[:-1]) + '.jpg'


class TrainDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path, crop_size=256):
        super(TrainDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = [f for f in os.listdir(hazy_path) if is_image_file(f)]
        self.hazy_image_list.sort()
        self.crop_size = crop_size

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = get_clear_name(hazy_image_name)
        
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)
        
        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')

        # 采用随机裁剪和旋转增强
        crop_params = RandomCrop.get_params(hazy, [self.crop_size, self.crop_size])
        rotate_params = random.randint(0, 3) * 90
        # 随机裁剪
        hazy = crop(hazy, *crop_params)
        clear = crop(clear, *crop_params)
        # 随机旋转
        hazy = rotate(hazy, rotate_params)
        clear = rotate(clear, rotate_params)
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)
        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)


class TestDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path=None):
        super(TestDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = [f for f in os.listdir(hazy_path) if is_image_file(f)]
        self.hazy_image_list.sort()
        if clear_path:
            self.clear_image_list = [f for f in os.listdir(clear_path) if is_image_file(f)]
            self.clear_image_list.sort()
        else:
            self.clear_image_list = None

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        if self.clear_image_list:
            clear_image_name = get_clear_name(hazy_image_name)
            clear_image_path = os.path.join(self.clear_path, clear_image_name)
            clear = Image.open(clear_image_path).convert('RGB')
            clear = to_tensor(clear)
            return hazy, clear, hazy_image_name
        else:
            return hazy, hazy_image_name

    def __len__(self):
        return len(self.hazy_image_list)


class ValDataset(data.Dataset):
    def __init__(self, hazy_path, clear_path):
        super(ValDataset, self).__init__()
        self.hazy_path = hazy_path
        self.clear_path = clear_path
        self.hazy_image_list = [f for f in os.listdir(hazy_path) if is_image_file(f)]
        self.clear_image_list = [f for f in os.listdir(clear_path) if is_image_file(f)]
        self.hazy_image_list.sort()
        self.clear_image_list.sort()

    def __getitem__(self, index):
        hazy_image_name = self.hazy_image_list[index]
        clear_image_name = get_clear_name(hazy_image_name)
        hazy_image_path = os.path.join(self.hazy_path, hazy_image_name)
        clear_image_path = os.path.join(self.clear_path, clear_image_name)
        hazy = Image.open(hazy_image_path).convert('RGB')
        clear = Image.open(clear_image_path).convert('RGB')
        to_tensor = ToTensor()
        hazy = to_tensor(hazy)
        clear = to_tensor(clear)
        return hazy, clear

    def __len__(self):
        return len(self.hazy_image_list)