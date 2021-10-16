# General utility files for preprocessing data
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import pydicom
import cv2

'''
Problem:
patient = each sample consisting a mini-batch
2D image = images averaged over z-axis (each channel = MRI type)
3D image = images stacked over z-axis (each channel = MRI type)
'''
# FLAIR Mean
# 13.533082209054646
# FLAIR std
# 31.19763992639374
# T1 mean
# 22.880566087409175
# T1 std
# 44.24645852278421
# T1CE mean 
# 10.282504520602227
# T1CE std
# 44.8414383440132

def preprocess(config):
    if config['type'] == 'pad':
        return transforms.Pad(**config['params'])
    elif config['type'] == 'resize':
        return transforms.Resize(**config['params'])
    elif config['type'] == 'randomcrop':
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


# RSNA-MICCAI Brain Tumor Radiogenomic Classification specific utils
def mri_png2array(image_path_dict,
                  output_type="2D"):
    '''
    :param image_path_dict:
    must be a dictionary with its keys correponding to each MRI type
    values corresponding to list of png image paths.
    :return:
    numpy array 3D stacked MRI scan
    '''
    
    # stacked_img = [[] for _ in image_path_dict]
    stacked_img = []
    for mri_i, each_MRI in enumerate(image_path_dict):
        for each_img_path in image_path_dict[each_MRI]:
            img = np.array(Image.open(each_img_path).convert("L"))
            stacked_img.append(img)
    
    if output_type == "2D":
        stacked_img = np.asarray(stacked_img)
        stacked_img = np.average(stacked_img, axis=1)
    elif output_type == "25D":
        
        stacked_img  = stacked_img
        
    return stacked_img


# Sampling scheme for MRI scans (i.e. how to sample along z-axis)
def random_stack(img_path_list,
                 N_samples=10):
    return np.random.choice(img_path_list, N_samples)


def sequential_stack(img_path_list,
                     N_samples=10):
    N = len(img_path_list)
    if N_samples > N:
        # Random sample additional images to match the number of samples
        add_samples = N_samples - N
        sampled = np.random.choice(img_path_list, add_samples).tolist()
        img_path_list = sorted(img_path_list + sampled)
        return img_path_list
    else:
        d = N // N_samples
        indices = range(0, N, d)[:N_samples]
        img_path_list = np.array(img_path_list)[indices]

        return img_path_list.tolist()


def process_label_csv(source_csv="./train_labels.csv",
                      target_csv="./experiments/exp1/train.csv",
                      K_fold=5,
                      seed=1234
                      ):
    df = pd.read_csv(source_csv)
    X = df["BraTS21ID"]
    y = df["MGMT_value"]
    skf = StratifiedKFold(n_splits=K_fold,
                          shuffle=True,
                          random_state=seed)
    flag_indices = np.zeros(len(y))

    for fold_i, (train_index, val_index) in enumerate(skf.split(X, y)):
        flag_indices[val_index] = fold_i

    df["BraTS21ID"] = df["BraTS21ID"].astype(str).str.zfill(5)
    df["flag_index"] = flag_indices.astype(int)
    df.to_csv(target_csv)

    return


def load_dicom(path, size):
    ''' 
    Reads a DICOM image, standardizes so that the pixel values are between 0 and 1, then rescales to 0 and 255
    
    Not super sure if this kind of scaling is appropriate, but everyone seems to do it. 
    '''
    dicom = pydicom.read_file(path)
    data = dicom.pixel_array
    # transform data into black and white scale / grayscale
#     data = data - np.min(data)
    if np.max(data) != 0:
        data = data / np.max(data)
    data = (data * 255).astype(np.uint8)
    return cv2.resize(data, (size, size))

import os 
import glob
from joblib import Parallel,delayed
import multiprocessing as mp

def get_all_image_paths(brats21id, image_type, folder='train'): 
    '''
    Returns an arry of all the images of a particular type for a particular patient ID
    '''
    # assert(image_type in mri_types)
    
    patient_path = os.path.join(
        "../rsna-miccai-brain-tumor-radiogenomic-classification/%s/" % folder, 
        str(brats21id).zfill(5),
    )
    
    paths = sorted(
        glob.glob(os.path.join(patient_path, image_type, "*")), 
        key=lambda x: int(x[:-4].split("-")[-1]),
    )
    
    num_images = len(paths)
    
    start = int(num_images * 0.35)
    end = int(num_images * 0.65)

    interval = 1
    
    if num_images < 10: 
        interval = 1

    return np.array(paths[start:end:interval])

def get_all_images(brats21id, image_type, folder='train', size=225,parell=True):
    if parell == True:
        return Parallel(n_jobs=mp.cpu_count(),prefer='threads')(delayed(load_dicom)(path, size) for path in get_all_image_paths(brats21id, image_type, folder))
    else:
        return [load_dicom(path, size) for path in get_all_image_paths(brats21id, image_type, folder)]

def get_image_plane(data):
    '''
    Returns the MRI's plane from the dicom data.
    
    '''
    x1,y1,_,x2,y2,_ = [round(j) for j in ast.literal_eval(data.ImageOrientationPatient)]
    cords = [x1,y1,x2,y2]

    if cords == [1,0,0,0]:
        return 'coronal'
    if cords == [1,0,0,1]:
        return 'axial'
    if cords == [0,1,0,0]:
        return 'sagittal'
