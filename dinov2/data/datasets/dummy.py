from torch.utils.data import Dataset
from torchvision.datasets.vision import VisionDataset
import torch
import numpy as np
from PIL import Image

class DummyDataset(Dataset):
    def __init__(self, shape=[1024,224,224,3], transforms=None, transform=None, target_transform=None, **kwargs):
        super().__init__('root', transforms, transform, target_transform)
        self.shape = shape # nsamples x chn x h x w

    def __len__(self):
        return self.shape[0]
    
    def get_image_data(self, index):
        return np.random.randn(*self.shape[1:])
    
    def get_target(self, index):
        return np.random.randint(0, 10)

    def __getitem__(self, index):
        image = np.random.randint(0,255, size=self.shape[1:], dtype=np.uint8)
        target = np.random.randint(0, 10)        
        image = Image.fromarray(image).convert('RGB')

        if self.transforms is not None:
            image, target = self.transforms(image, target)
        # print(f'global crops: {len(image["global_crops"])} x {image["global_crops"][0].shape}'), 
        # print(f'local crops: {len(image["local_crops"])} x {image["local_crops"][0].shape}'), 
        return image, target