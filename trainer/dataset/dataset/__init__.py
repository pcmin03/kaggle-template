import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pandas as pd
from .data_utils import preprocess
from .vocsegment import myVOCSegmentation
from .endoscopy_dataset import EndoscopyData

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