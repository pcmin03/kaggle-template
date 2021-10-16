import torch
import logging

LOGGER = logging.getLogger(__name__)

def create(conf, model):

    if conf['type'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), **conf['params'])
    elif conf['type'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), **conf['params'])
    elif conf['type'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), **conf['params'], weight_decay=0.008)
    else:
        raise AttributeError(f'not support optimizer config: {conf}')

    return optimizer