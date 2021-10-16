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
from glob import glob
from joblib import Parallel,delayed
import multiprocessing as mp
import nibabel as nib
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

class custom_dataset_nifti(torch_data.Dataset):
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
        
        # self.transform = tio.OneOf({
        #                 tio.RandomAffine(): 0.75,
        #             })
        self.transform = transforms.ToTensor()

        # self.sampling_scheme = {"random": random_stack,
        #                         "sequential": sequential_stack}[conf["sampling_scheme"]]
        
        images_paths,masks_path = self.preload_img_paths()
        images= self.get_all_images(images_paths,size=256)
        maskes= self.get_all_images(masks_path,size=256)

        crop_images=[]
        stack_label = []
        
        for i,j,z in zip(images,maskes,targets):
            filterdim = self.crop_dim(j)
            crop_img = i[:,:,filterdim]
#             crop_mask = j[:,:,filterdim]>0
#             crop_img = np.where(crop_mask,crop_img,crop_mask)

            crop_images.append(crop_img)
            stack_label.append(np.tile(z,len(crop_img[0,0,:])))

        # print(np.concatenate(crop_images,axis=-1).shape)
        self.crop_images = np.concatenate(crop_images,axis=-1).transpose((2,0,1))
        self.stack_label = np.concatenate(stack_label,axis=-1)
        print(self.crop_images.shape,self.stack_label.shape)
    def __len__(self):
        return len(self.stack_label)

    def crop_dim(self,voxel):
        if voxel.sum() == 0:
            return voxel
        keep = (voxel.mean(axis=(0, 1)) > 1)
        return keep

    def read_niifile(self,niifile,size=256):  # Read niifile file
        img = nib.load(niifile)  # Download niifile file (actually extract the file)
        img_fdata = img.get_fdata()  # Get niifile data

        if np.max(img_fdata) != 0:
            img_fdata = img_fdata / np.max(img_fdata)
        img_fdata = (img_fdata * 255).astype(np.uint8)
        
        return cv2.resize(img_fdata,(size,size))
        
    def get_all_images(self,img_list, folder='train', size=225):
        return Parallel(n_jobs=mp.cpu_count(),prefer='threads')(delayed(self.read_niifile)(path, size) for path in tqdm(img_list))


    def preload_img_paths(self):
        mri_types = self.conf["mri_types"]
        
        patientes = []
        patientes_mask = []
        for imgpath in tqdm(self.input_paths,desc='loading path'):
            # patient_type = {}
            # seg_mask_type = {}
            patient_type = []
            seg_mask_type = []
            
            pid = imgpath.split('/')[-1]
            # for j in mri_types:
               
                # patient_type[j] = glob(f'{imgpath}_{j}_*nii.gz')[0]
                # seg_mask_type[j] = glob(f'/nfs3/personal/cmpark/project/kaggle/challenge/rsna-preprocessed/mask_img/{pid}.nii.gz')[0]
                # print(patient_type,seg_mask_type)
            # patient_type.append(glob(f'{imgpath}_{j}_*nii.gz')[0])
            # seg_mask_type.append(glob(f'/nfs3/personal/cmpark/project/kaggle/challenge/rsna-preprocessed/mask_img/{pid}.nii.gz')[0])
            # patientes.append(glob(f'{imgpath}_{mri_types[0]}_*nii.gz')[0])
            patientes.append(glob(f'/nfs3/personal/cmpark/project/kaggle/challenge/rsna-preprocessed/train/{pid}/T1wCE/T1wCE.nii.gz')[0])
            
            # patientes.append(glob(f'/nfs3/personal/cmpark/project/kaggle/challenge/rsna-preprocessed/sample/{pid}_0001.nii.gz')[0])
            patientes_mask.append(glob(f'/nfs3/personal/cmpark/project/kaggle/challenge/rsna-preprocessed/mask_img/{pid}.nii.gz')[0])
            
            # patientes.append(patient_type)
            # patientes_mask.append(seg_mask_type)
        return patientes,patientes_mask
# rsna-miccai-brain-tumor-radiogenomic-classification_nifti/train/00803_T1wCE_0.nii.gz'
    # def load_input(self, index):
    #     sampled_imgs = self.sampled_image_paths[index]
    #     inputs = []
    #     zdim = random.randrange(20,30)
    #     for i in self.conf["mri_types"]:
    #         proxy = nib.load(sampled_imgs[i][0])
    #         arr = proxy.get_fdata()
    #         arr += arr.min()
    #         inputs.append(proxy.get_fdata()[:,:,zdim:zdim+20])
        
        # for mri_i, each_MRI in enumerate(sampled_imgs):
        #     if len(sampled_imgs[each_MRI]) < 11: 
        #         print(sampled_imgs[each_MRI],each_MRI,len(sampled_imgs[each_MRI]),len(sampled_imgs[each_MRI])-self.conf['N_sample'])
        #     start = random.randint(0,len(sampled_imgs[each_MRI])-self.conf['N_sample'])
        #     sampled_imgs[each_MRI] = sampled_imgs[each_MRI][start : start + self.conf['N_sample']]
        
        # inputs = mri_png2array(sampled_imgs,
        #                        output_type=self.conf["output_type"])
        
        # return np.concatenate(inputs,axis=2)[None]

    def __getitem__(self, index):
        inputs = self.crop_images[index]
        
        if self.mode == 'train': 
            inputs = self.transform(inputs)
        else: 
            inputs = self.transform(inputs)

        # if self.conf['output_type'] == '25D' or self.conf['output_type'] == '3D':
        #     inputs = inputs.transpose(2,0,1)
            # inputs = torch.Tensor(inputs)
        # print(inputs.type())
        
        if self.mode != "test":
            return inputs, self.stack_label[index]
        else:
            return inputs