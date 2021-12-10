import os
import random
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from PIL import Image


class AnimeDataset(Dataset):
    def __init__(self, root_path):
        super().__init__()

        self.image_size = (256, 256)

        image_folder = os.path.join(root_path, 'real')
        smooth_folder = os.path.join(root_path, 'smooth')
        style_folder = os.path.join(root_path, 'style')

        self.image_path_list = [os.path.join(image_folder, file_name) for file_name in os.listdir(image_folder)]
        self.style_path_list = [os.path.join(smooth_folder, file_name) for file_name in os.listdir(smooth_folder)]
        self.smooth_path_list = [os.path.join(style_folder, file_name) for file_name in os.listdir(style_folder)]

        self.transforms = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
            transforms.Normalize(0.5, 0.5),
        ])

    def __len__(self):
        return len(self.style_path_list)

    def __getitem__(self, index):
        image_path = random.choice(self.image_path_list)
        style_path = random.choice(self.style_path_list)
        smooth_path = random.choice(self.smooth_path_list)

        image = Image.open(image_path).convert('RGB')
        gray = image.convert('L').convert('RGB')
        style = Image.open(style_path).convert('RGB')
        smooth = Image.open(smooth_path).convert('RGB')

        image = self.transforms(image)
        gray = self.transforms(gray)
        style = self.transforms(style)
        smooth = self.transforms(smooth)

        inputs = {
            'image': image,
            'gray': gray,
            'style': style,
            'smooth': smooth,
        }

        return inputs


if __name__ == "__main__":
    import torch

    anime_dataset = AnimeDataset(root_path='./data')

    inputs = anime_dataset[0]

    image = inputs['image'].unsqueeze(0)
    gray = inputs['gray'].unsqueeze(0)
    style = inputs['style'].unsqueeze(0)
    smooth = inputs['smooth'].unsqueeze(0)


    torchvision.utils.save_image(torch.cat([image, gray, style, smooth]), './test.png')