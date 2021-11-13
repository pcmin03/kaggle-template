# General utility files for preprocessing data
from torchvision import transforms
from PIL import Image
import numpy as np
from glob import glob
import pandas as pd
from sklearn.model_selection import StratifiedKFold
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
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
