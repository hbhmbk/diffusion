import torch
from torch.utils.data import Dataset
from utils import *
from PIL import Image
import utils
from torchvision import transforms
path='E:\\vqvae\diffusion\\anime-faces'
class MyDatasset(Dataset):
    def __init__(self,images_path : list,transform=None):
        self.images_path=images_path
        self.transform=transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))

        if self.transform is not None:
            img = self.transform(img)
        return img



