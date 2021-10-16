import torch.utils.data as torch_data
import sklearn.model_selection as sk_model_selection
import pandas as pd
from torchvision import transforms
from glob import glob
import numpy as np
from .data_utils import mri_png2array, random_stack, sequential_stack
from tqdm import tqdm 
import os 
import random
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from .data_utils import * 
import nibabel as nib
import torchio as tio
from .data_utils import load_dicom,get_all_images



class custom_dataset_dicom(torch_data.Dataset):
    def __init__(self,
                 input_paths,
                 mode="train",
                 transform=transforms.ToTensor(),
                 conf=None):
        '''
        targets: list of values for classification
        or list of paths to segmentation mask for segmentation task.
        augment: list of keywords for augmentations.
        '''
        
        
        self.mode = mode
        self.conf = conf
        self.input_paths = input_paths
        self.transform = transforms.ToTensor()
        
        self.images,self.labels,self.img_ids = self.get_all_data()
        
        # Parallel(n_jobs=mp.cpu_count(),prefer='threads')(delayed(self.get_all_data()) for path in get_all_image_paths(brats21id, image_type, folder))

        # self.sampled_image_paths = self.sampled_image_paths
    def __len__(self):
        return len(self.images)

    
    def get_all_data(self, image_size=128):
        
        X = []
        y = []
        train_ids = []

        for i in tqdm(self.input_paths.index):
            x = self.input_paths.loc[i]
        
            for mri_type in self.conf['mri_types']:
                
                
                images = get_all_images(int(x['BraTS21ID']), mri_type, 'train', image_size)
                label = x['MGMT_value']

            X += images
            y += [label] * len(images)
            train_ids += [int(x['BraTS21ID'])] * len(images)
            assert(len(X) == len(y))
        return np.array(X), np.array(y), np.array(train_ids)


    def __getitem__(self, index):
        inputs = self.images[index]
        
        if self.mode == 'train': 
            inputs = self.transform(inputs)
        else: 
            inputs = self.transform(inputs)
        
        if self.conf['output_type'] == '25D':
            # inputs = inputs.transpose(2,0,1)
            inputs = inputs
        elif self.conf['output_type'] =='3D':
            inputs = inputs[None]
            # inputs = torch.Tensor(inputs)
        # print(inputs.type())
        
        if self.mode != 'test': 
            targets = self.labels[index]    
            return inputs, targets
        else : 
            targets = self.labels[index]    
            return inputs, targets,self.img_ids[index]

    