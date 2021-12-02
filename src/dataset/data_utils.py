# General utility files for preprocessing data
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
import cv2

def ratio2cordi(imgshape,cordi): 
    w,h = imgshape.shape[:2]
    cordi = cordi / 100 
    cordi[:,0] *= h
    cordi[:,1] *= w
    return cordi

def cordi2mask(cordi): 
    cordi = Polygon(cordi)
    return 

def json2df(json_data): 
    label = ['id','imgpath','Atrophy','IM','Biopsy','Etc','atrophy_grade','im_grade','comment']
    df = pd.DataFrame(columns=label)
    
    imgpath = json_data['data']['image'].replace('/data','.')
    df['imgpath'] = [imgpath]
    df['id'] = [json_data['id']]
    annoes = json_data['completions'][0]['result']
    texture = []
    polygones = []
    for cordi in annoes: 
        labeltype = cordi['from_name']
        labelvalue = cordi['value']
        
        if labeltype == 'class_label': # put to corrdination
            df[labelvalue['polygonlabels'][0]] = [labelvalue['points']]
#             polygones.append(cordi['value']['points'])        
        elif labeltype in ['atrophy_grade','im_grade']: # put to scoring
            df[labeltype] = np.array(labelvalue['text'][0])
            
        elif labeltype in 'comment': 
            df[labeltype] = [labelvalue['text']]
        else:
            assert f'{labeltype} : different label type'
    return df

def preprocess(config):
    
    if config['type'] == 'randomcrop':
        return transforms.RandomCrop(**config['params'])
    elif config['type'] == 'horizontal':
        return transforms.RandomHorizontalFlip()
    elif config['type'] == 'tensor':
        return transforms.ToTensor()
    elif config['type'] == 'normalize':
        return transforms.Normalize(**config['params'])

def load_png(image_path, channel_type="L"):  # "RGB" or "L"
    # 2D
    return np.array(Image.open(image_path).convert(channel_type))

# Sampling scheme for MRI scans (i.e. how to sample along z-axis)
def random_stack(img_path_list,
                 N_samples=10):
    return np.random.choice(img_path_list, N_samples)

def create_folds(df, n_s=5, n_grp=None):
    df['kfold'] = -1
    
    if n_grp is None:
        skf = KFold(n_splits=n_s)
        target = df['Pawpularity']
    else:
        skf = StratifiedKFold(n_splits=n_s, shuffle=True)
        df['grp'] = pd.cut(df['Pawpularity'], n_grp, labels=False)
        target = df.grp
    
    for fold_no, (t, v) in enumerate(skf.split(target, target)):
        df.loc[v, 'kfold'] = fold_no

    df = df.drop('grp', axis=1)
    
    return df
