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

class EndoscopyData(Dataset):
    def __init__(self,root,transform,image_set,biopsy_loc):
        file_dir = root
        
        with open(file_dir) as json_file: 
            json_data = json.load(json_file) # complate file
            totaldf = []
            for i in tqdm(json_data): 
                totaldf.append(json2df(i))
            totaldf = pd.concat(totaldf)
        totaldf = totaldf.sort_values('id')
        self.totaldf = self.prerpocess(totaldf)
        self.random_crop = A.RandomCrop(width=512, height=512)
        self.valid_random_crop = A.RandomCrop(width=800, height=800)
        self.transforms = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
        ])
        
        if biopsy_loc == 'Antrum':
            self.totaldf = self.totaldf[self.totaldf[biopsy_loc] == 1]
        elif biopsy_loc == 'Body_LC':
            self.totaldf = self.totaldf[self.totaldf[biopsy_loc] == 1]
        print(len(self.totaldf))
        # kfold
        self.mode = image_set
        train_idx,test_idx,_,_ = self.split_train_test(self.totaldf['id'].to_numpy(),self.totaldf['atrophy_grade'].to_numpy())

        if image_set == 'train': 
            self.totaldf = self.totaldf[self.totaldf['id'].isin(train_idx)]
        else : 
            self.totaldf = self.totaldf[self.totaldf['id'].isin(test_idx)]

        # _,class_loc = np.unique(self.totaldf['atrophy_grade'].to_numpy(),return_inverse=True) # upsampling
        oversampledf = self.totaldf[self.totaldf['atrophy_grade'] == '3']
        self.totaldf = self.totaldf.append(oversampledf)
        print(np.unique(self.totaldf['atrophy_grade'].to_numpy(),return_counts=True))

    def split_train_test(self,x,y):
        stk = StratifiedKFold(n_splits=3, random_state=None, shuffle=False)
        
        for num,(train_index, test_index) in enumerate(stk.split(x,y)):
            if num == 0 :
                return x[train_index], x[test_index],y[train_index], y[test_index]
            else : 
                assert 'Too large kfold number'

    def class_labeling(self,grade): 
        if grade == '3' or grade == '1' :
            return 1
        elif grade == '0':
            return 0
        else :
            return 0

    def prerpocess(self,df): 
        df = df[df['Atrophy'].notna()]
        df['Antrum'] = df.comment.apply(lambda x : 1 if 'Antrum' in np.array(x) else 0)
        df['Body_LC'] = df.comment.apply(lambda x : 1 if 'Body LC' in np.array(x) else 0)
        df['imgpath'] = df.imgpath.apply(lambda x : x.replace('./upload','/ssd1/cmpark/tmp_endoscopy_images/NCC/upload'))
        df = df[~df['atrophy_grade'].isna()]
        
        return df

    def __len__(self): 
        return len(self.totaldf)

    def __getitem__(self, index):
        sample = self.totaldf.iloc[index]
        class_num = self.class_labeling(sample['atrophy_grade'])
        
        img = cv2.imread(sample[1])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        annot = ratio2cordi(img,np.array(sample['Atrophy']))
        annot = annot.astype(np.int32)
        target = cv2.fillPoly(np.zeros_like(img[:,:,0]),[annot],class_num)

        if self.mode == 'train': 
            if self.transforms is not None:
                
                augmented = self.transforms(image=np.array(img), mask=np.array(target))
                augmented = self.random_crop(image=augmented['image'],mask=augmented['mask'])
                
                img = augmented['image']
                target = augmented['mask']
        else:
            augmented = self.valid_random_crop(image=np.array(img), mask=np.array(target))
            img = augmented['image']
            target = augmented['mask']

        img = to_tensor(img)
        # target = (np.arange(2) == target[...,None]-1).astype(int) #2d one hot endoing
        target = torch.from_numpy(target)
        
        return img, target