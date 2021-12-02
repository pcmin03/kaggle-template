from torchvision.transforms.functional import to_tensor
from torch.utils.data import Dataset
import torch
import numpy as np
from skimage.segmentation import mark_boundaries
import matplotlib.pylab as plt
from albumentations import HorizontalFlip, Compose, Resize, Normalize
import os
import time
from .data_utils import json2df,ratio2cordi,cordi2mask
import cv2 
import json
from tqdm import tqdm
import pandas as pd
import albumentations as A
from sklearn.model_selection import StratifiedKFold
import torch.nn.functional as F
from sklearn.preprocessing import OneHotEncoder
from albumentations.pytorch.transforms import ToTensorV2

class PawpularityDataset(Dataset):
    def __init__(self, root_dir, df, mode,conf):
        trans_config = conf['transform']
        transforms = {
            "train": A.Compose([
                A.Resize(trans_config['img_size'], trans_config['img_size']),
                A.HorizontalFlip(p=0.5),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.),
            
            "valid": A.Compose([
                A.Resize(trans_config['img_size'], trans_config['img_size']),
                A.Normalize(
                        mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225], 
                        max_pixel_value=255.0, 
                        p=1.0
                    ),
                ToTensorV2()], p=1.)
        }


        self.root_dir = root_dir
        self.df = df
        self.file_names = df['file_path'].values
        self.targets = df['Pawpularity'].values
        self.transforms = transforms[mode]
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, index):
        
        img_path = str(self.file_names[index])
        
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        target = self.targets[index]
        
        if self.transforms:
            img = self.transforms(image=img)["image"]
            
        return img, target