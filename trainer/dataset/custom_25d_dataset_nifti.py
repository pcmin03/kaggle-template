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


class custom_25d_dataset_nifti(torch_data.Dataset):
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
        # self.transform = A.Compose([
        #                 A.Resize(192,192),
        #                 A.CLAHE(p=0.5),

        #                 ])
        # self.totensor = ToTensorV2() 
        
        self.transform = tio.OneOf({
                        tio.RandomAffine(): 0.75,
                    })

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
                patient_type[j] = glob(f'{imgpath}_{j}_*nii.gz')            
            patientes.append(patient_type)
            
        return patientes

    def load_input(self, index):
        sampled_imgs = self.sampled_image_paths[index]
        inputs = []
        zdim = random.randrange(20,30)
        for i in self.conf["mri_types"]:
            proxy = nib.load(sampled_imgs[i][0])
            arr = proxy.get_fdata()
            arr += arr.min()
            inputs.append(proxy.get_fdata()[:,:,zdim:zdim+20])
        
        # for mri_i, each_MRI in enumerate(sampled_imgs):
        #     if len(sampled_imgs[each_MRI]) < 11: 
        #         print(sampled_imgs[each_MRI],each_MRI,len(sampled_imgs[each_MRI]),len(sampled_imgs[each_MRI])-self.conf['N_sample'])
        #     start = random.randint(0,len(sampled_imgs[each_MRI])-self.conf['N_sample'])
        #     sampled_imgs[each_MRI] = sampled_imgs[each_MRI][start : start + self.conf['N_sample']]
        
        # inputs = mri_png2array(sampled_imgs,
        #                        output_type=self.conf["output_type"])
        
        return np.concatenate(inputs,axis=2)[None]

    def load_target(self, index):  # For classification task
        return self.targets[index]

    def __getitem__(self, index):
        inputs = self.load_input(index)
        
        inputs = (self.transform(inputs))[0]
        
        if self.conf['output_type'] == '25D' or self.conf['output_type'] == '3D':
            inputs = inputs.transpose(2,0,1)
            # inputs = torch.Tensor(inputs)
        # print(inputs.type())
        
        if self.mode != "test":
            targets = self.load_target(index)
            return inputs, targets
        else:
            return inputs