import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pandas as pd
from pathlib import Path
from .data_utils import preprocess,create_folds
from .vocsegment import myVOCSegmentation
from .endoscopy_dataset import EndoscopyData
from .petfinder_dataset import PawpularityDataset
# from .ventilator_data import Ventilator_Dataset
LOGGER = logging.getLogger(__name__)


def create(conf, mode='train'):
    
    conf = conf[mode]
    # transformers = transforms.Compose([preprocess(t) for t in conf['preprocess']] )
    if conf['name'] == 'voc':
        dataset = myVOCSegmentation(
            root='../', 
            download=False, 
            transform=None, 
            image_set= 'val'
            )
    
    elif conf['name'] == 'endoscopy': 
        dataset = EndoscopyData(
            root=conf['datapath'], 
            transform=conf['transform'], 
            image_set=mode,
            biopsy_loc = conf['biopsy']
            )
    elif conf['name'] == 'petfinder': 
        base_path = Path(conf['datapath'])
        df = pd.read_csv(base_path/"train.csv")
        df['file_path'] = df['Id'].apply(lambda x : base_path / conf['train_dir']/ f'{x}.jpg')
        feature_cols = [col for col in df.columns if col not in ['Id', 'Pawpularity', 'file_path']]

        df = create_folds(df, n_s=5, n_grp=14)
        if mode == 'train': 
            df_data = df[df.kfold != conf['fold']].reset_index(drop=True)    
        elif mode == 'valid': 
            df_data = df[df.kfold == conf['fold']].reset_index(drop=True)
            
        dataset = PawpularityDataset(base_path / conf['train_dir'],df_data,mode,conf)
            
    else:
        raise AttributeError(f'not support dataset config: {conf}')
    
    # sampler = torch.utils.data.distributed.DistributedSampler(
    #         dataset, 
    #         num_replicas=world_size, 
    #         rank=local_rank,
    #         shuffle= True if mode == 'train' else False
    #     )
    
    return dataset

#     requirement_whl/efficientnet_pytorch-0.6.3.tar.gz
# requirement_whl/pretrainedmodels-0.7.4.tar.gz
# requirement_whl/segmentation_models_pytorch-0.2.0-py3-none-any.whl