from torchvision.datasets import VOCSegmentation
from torchvision.transforms.functional import to_tensor, to_pil_image
from PIL import Image
import torch
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as plt
from albumentations import HorizontalFlip, Compose, Resize, Normalize
import os
import time

class myVOCSegmentation(VOCSegmentation):
    def __getitem__(self, index):
        img = Image.open(self.images[index]).convert('RGB')
        target = Image.open(self.masks[index])

        if self.transforms is not None:
            augmented = self.transforms(image=np.array(img), mask=np.array(target))
            img = augmented['image']
            target = augmented['mask']
            target[target>20] = 0
        
        img = to_tensor(img)
        target = torch.from_numpy(target).type(torch.long)
        return img, target