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

# def preprocess(config):
#     if config['type'] == 'pad':
#         return A.Pad(**config['params'])
#     elif config['type'] == 'resize':
#         return A.Resize(**config['params'])
#     elif config['type'] == 'randomcrop':
#         return A.RandomCrop(**config['params'])
#     elif config['type'] == 'horizontal':
#         return A.RandomHorizontalFlip()
#     elif config['type'] == 'tensor':
#         return transforms.ToTensor()
#     elif config['type'] == 'normalize':
#         return transforms.Normalize(**config['params'])

class custom_25d_dataset_nifti_smalldata(torch_data.Dataset):
    def __init__(self,
                 input_paths,
                 targets=None,
                 mode="train",
                 transform=transforms.ToTensor(),
                 conf=None):
        '''
        targets: list of values for classification
        or list of paths to segmentation mask for segmentation task.
        augment: list of keywords for augmentations.
        '''
        
        self.input_paths = input_paths  # paths to patients
        self.targets = targets
        self.mode = mode
        self.transform = transform
        self.conf = conf

        self.transform = tio.OneOf({
                        tio.RandomAffine(): 0.75,
                    })
        self.preprocessing = tio.Compose([
                    tio.Clamp(0),
                    tio.ZNormalization()
                ])

        self.spatial = tio.Compose([
                        tio.RandomElasticDeformation(num_control_points=(7, 7, 7),locked_borders=2,p=0.5),
                        tio.RandomAnisotropy(axes=(0, 1, 2),p=0.5),
                        tio.RandomAffine(0.3,p=0.5),
                        tio.RandomSwap((10,10,10),p=0.5),
                        tio.RandomFlip(axes=('LR',),p=0.5)])
                        
        # 38.07043296258841 47.407259928178
        # 54.58037165415802 63.68226527510233
        # 28.78268175394762 34.48558789437326
        # 33.892640064086 42.87205675912448

        self.sampled_image_paths = self.preload_img_paths()
        # self.sampled_image_paths = self.sampled_image_paths
    def __len__(self):
        return len(self.sampled_image_paths)

    def preload_img_paths(self):
        mri_types = self.conf["mri_types"]
        
        patientes = []
        for imgpath in tqdm(self.input_paths,desc='loading path'):
            patient_type = {}
            for j in mri_types:
                patient_type[j] = f'{imgpath}/{j}.npy'
            patientes.append(patient_type)
            
        return patientes

    def load_input(self, index):
        sampled_imgs = self.sampled_image_paths[index]
        inputs = []
        zdim = random.randrange(20,60)
        for i in self.conf["mri_types"]:
            inputs.append(np.load(sampled_imgs[i])[:,:,zdim:zdim+self.conf['N_sample']])
            
        return np.concatenate(inputs,axis=2)[None]

    def load_target(self, index):  # For classification task
        return self.targets[index]

    def __getitem__(self, index):
        inputs = self.load_input(index)
        
        if self.mode == 'train': 
            inputs = (self.spatial(self.preprocessing(inputs/255.)))[0]
        elif self.mode == 'valid': 
            inputs = self.preprocessing(inputs/255.)[0]
        
        if self.conf['output_type'] == '25D':
            inputs = inputs.transpose(2,0,1)
            # inputs = torch.Tensor(inputs)
        # print(inputs.type())
        
        if self.mode != "test":
            targets = self.load_target(index)
            return inputs, targets
        else:
            return inputs