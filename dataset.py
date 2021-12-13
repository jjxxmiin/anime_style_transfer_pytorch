import os
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class AnimeDataset(Dataset):
    def __init__(self, root_path, dataset):
        super().__init__()

        self.image_size = (256, 256)

        image_folder = os.path.join(root_path, 'pixiv', 'real')
        smooth_folder = os.path.join(root_path, dataset, 'smooth')
        style_folder = os.path.join(root_path, dataset, 'style')

        self.image_path_list = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]
        self.smooth_path_list = [os.path.join(smooth_folder, file_name) for file_name in os.listdir(smooth_folder)]
        self.style_path_list = [os.path.join(style_folder, file_name) for file_name in os.listdir(style_folder)]

        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, idx):
        image_path = random.choice(self.image_path_list)
        style_path = self.style_path_list[idx] # random.choice(self.style_path_list)
        smooth_path = self.smooth_path_list[idx] # random.choice(self.smooth_path_list)

        image = Image.open(image_path).convert('RGB')
        anime = Image.open(style_path).convert('RGB')
        anime_gray = Image.open(style_path).convert('L').convert('RGB')
        anime_smooth = Image.open(smooth_path).convert('L').convert('RGB')

        image = self.transforms(image)
        anime = self.transforms(anime)
        anime_gray = self.transforms(anime_gray)
        anime_smooth = self.transforms(anime_smooth)

        inputs = {
            'image': image,
            'anime': anime,
            'anime_gray': anime_gray,
            'anime_smooth': anime_smooth,
        }

        return inputs