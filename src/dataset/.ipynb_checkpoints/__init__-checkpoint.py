import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pandas as pd
from .data_utils import preprocess
from .ventilator_data import Ventilator_Dataset
LOGGER = logging.getLogger(__name__)


def create(conf, world_size=1, local_rank=-1, mode='train'):

    conf = conf[mode]
    transformers = transforms.Compose([preprocess(t) for t in conf['preprocess']] )

    if conf['name'] == 'mnist':
        dataset = torchvision.datasets.FashionMNIST(
            root='../', 
            download=True, 
            transform=transformers, 
            train= True if mode == 'train' else False
            )

    elif conf['name'] == 'ventil': 
        path = conf.basepath
        train = pd.read_csv(os.path.join(path, 'train.csv'))
        test = pd.read_csv(os.path.join(path, 'test.csv'))
        sub = pd.read_csv(os.path.join(path, 'sample_submission.csv'))

        dataset = Ventilator_Dataset(
            traindf = train
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