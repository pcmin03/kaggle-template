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

class cusom_25d_dataset(torch_data.Dataset):
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
        self.transform = A.Compose([
                        A.Resize(192,192),
                        A.CLAHE(p=0.5),

                        ])
        self.totensor = ToTensorV2() 
        
        # self.sampling_scheme = {"random": random_stack,
        #                         "sequential": sequential_stack}[conf["sampling_scheme"]]
        
        self.sampled_image_paths = self.preload_img_paths()
    def __len__(self):
        return len(self.sampled_image_paths)

    def preload_img_paths(self):
        mri_types = self.conf["mri_types"]
        
        patientes = []
        for imgpath in tqdm(self.input_paths,desc='loading path'):
            patient_type = {}
            for j in mri_types:
                patient_type[j] = glob(f'{imgpath}/{j}/*.png')            
            patientes.append(patient_type)
            
        return patientes

    def load_input(self, index):
        sampled_imgs = self.sampled_image_paths[index]
        for mri_i, each_MRI in enumerate(sampled_imgs):
            if len(sampled_imgs[each_MRI]) < 11: 
                print(sampled_imgs[each_MRI],each_MRI,len(sampled_imgs[each_MRI]),len(sampled_imgs[each_MRI])-self.conf['N_sample'])
            start = random.randint(0,len(sampled_imgs[each_MRI])-self.conf['N_sample'])
            sampled_imgs[each_MRI] = sampled_imgs[each_MRI][start : start + self.conf['N_sample']]
        
        inputs = mri_png2array(sampled_imgs,
                               output_type=self.conf["output_type"])
        
        return inputs

    def load_target(self, index):  # For classification task
        return self.targets[index]

    def __getitem__(self, index):
        inputs = self.load_input(index)
        
        if self.conf['output_type'] == '25D' or self.conf['output_type'] == '3D':
            # sample = self.transform(image=np.array(inputs[0]))['image']
            
            inputs = np.array([self.transform(image=np.array(i))['image'] for i in inputs])
            inputs = self.totensor(image=inputs/255.)['image'].permute(1,0,2)
            # inputs = torch.Tensor(inputs)
        # print(inputs.type())
        if self.mode != "test":
            targets = self.load_target(index)
            return inputs, targets
        else:
            return inputs