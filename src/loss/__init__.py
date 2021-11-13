import torch.nn as nn
import logging
from .MVL import MeanVarianceLoss

LOGGER = logging.getLogger(__name__)

def create(conf):

    if conf['type'] == 'ce':
        criterion = nn.CrossEntropyLoss(ignore_index=conf['ignore_index'])
    elif conf['type'] ==  'bce': 
        criterion = nn.BCEWithLogitsLoss()
    elif conf['type'] ==  'sL1loss': 
        criterion = nn.SmoothL1Loss()
    elif conf['type'] ==  'MAE': 
        criterion = nn.L1Loss()
    elif conf['type'] ==  'MSE': 
        criterion = nn.MSELoss()
    elif conf['type'] ==  'MVL': 
        criterion = MeanVarianceLoss(conf['lambda_1'],conf['lambda_2'],conf['start_age'],conf['end_age'])
    else:
        raise AttributeError(f'not support loss config: {conf}')

    return criterion