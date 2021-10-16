import torch
import logging

from warmup_scheduler import GradualWarmupScheduler
LOGGER = logging.getLogger(__name__)

def create(conf, optimizer):
    scheduler = None
    if conf['type'] == 'CosineAnnealingLR':
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **conf['params'])
    elif conf['type'] == 'CosineAnnealingWarmRestarts':
        scheduler_lr = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **conf['params'])
    else:
        raise AttributeError(f'not support scheduler config: {conf}')

    if conf['warmup']['status'] is True: 
        scheduler = GradualWarmupScheduler(optimizer, after_scheduler=scheduler_lr, **conf['warmup']['params'])
    
    else:
        scheduler = scheduler_lr

    return scheduler