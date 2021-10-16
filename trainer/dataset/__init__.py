import torch
import torchvision
import torchvision.transforms as transforms
import logging
import os
import pandas as pd
from .custom_dataset import custom_dataset
from .custom_25d_dataset import cusom_25d_dataset
from .custom_25d_dataset_nifti import custom_25d_dataset_nifti
from .data_utils import preprocess
from .custom_25d_dataset_nifti_smalldata import custom_25d_dataset_nifti_smalldata
from .custom_dataset_dicom import custom_dataset_dicom
from .custom_dataset_nifti import custom_dataset_nifti

LOGGER = logging.getLogger(__name__)


def create(conf, world_size=1, local_rank=-1, mode='train'):

    conf = conf[mode]
    transformers = transforms.Compose([preprocess(t) for t in conf['preprocess']] )

    if conf['name'] != 'None':
        if conf['name'] == 'mnist':
            dataset = torchvision.datasets.FashionMNIST(
                root='../', 
                download=True, 
                transform=transformers, 
                train=(mode == 'train')
                )
        else:
            df = pd.read_csv(conf['label_dir'])
            filerlist = [109,123,709]
            df = df[~df[conf['patient_id']].isin(filerlist)]
            
            flag_index = conf["flag_index"]   
            df = df[df["flag_index"].isin(flag_index)]
            targets = df[conf["label_name"]]
            
            input_paths = df[conf["patient_id"]]
            input_paths = [f'{conf["data_path"]}/{str(i).zfill(5)}' for i in input_paths]
            # input_paths = list(map(lambda x:f'{conf["data_path"]}/{x}'),input_paths)
            if conf['name'] == 'custom_dataset_dicom':
                dataset = custom_dataset_dicom(input_paths=df,
                                        mode=mode,
                                        transform=transformers,
                                        conf=conf
                                        )

            elif conf['name'] == 'custom_25d_dataset':
                dataset = cusom_25d_dataset(input_paths=input_paths,
                                        targets=targets,
                                        mode=mode,
                                        transform=transformers,
                                        conf=conf
                                        )
            elif conf['name'] == 'custom_25d_dataset_nifti':
                dataset = custom_25d_dataset_nifti(input_paths=input_paths,
                                        targets=targets,
                                        mode=mode,
                                        transform=transformers,
                                        conf=conf
                                        )
            elif conf['name'] == 'custom_25d_dataset_nifti_smalldata':
                dataset = custom_25d_dataset_nifti_smalldata(input_paths=input_paths,
                                        targets=targets,
                                        mode=mode,
                                        transform=transformers,
                                        conf=conf
                                        )
            elif conf['name'] == 'custom_dataset_nifti':
                dataset = custom_dataset_nifti(input_paths=input_paths,
                                        targets=targets,
                                        mode=mode,
                                        transform=transformers,
                                        conf=conf
                                        )

                                        
    else:
        raise AttributeError(f'not support dataset config: {conf}')
    
    sampler = torch.utils.data.distributed.DistributedSampler(
            dataset, 
            num_replicas=world_size, 
            rank=local_rank,
            shuffle=(mode == 'train')
        )
    
    dl = torch.utils.data.DataLoader(
        dataset,
        batch_size=conf['batch_size'],
        shuffle=False,
        pin_memory=True,
        drop_last=conf['drop_last'],
        num_workers=4,
        sampler=sampler
    )
    
    return dl, sampler