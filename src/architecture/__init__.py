import torch
import logging
import segmentation_models_pytorch as smp

from torch import nn
from . import toy
from . import custom_unet
from timm.models import create_model

LOGGER = logging.getLogger(__name__)
def create(conf, num_classes=None):

    # use only encoder    
    if conf['decoder']['status'] != True:
        base, architecture_name = [l.lower() for l in conf['type'].split('/')]
        if base == 'toy':
            architecture = toy.ToyModel()
        
        elif base == 'pawpularitymodel'
            architecture = PawpularityModel(base,architecture_name)
            
        else : 
            architecture = create_model(base,pretrained=conf['pretrained'])
    # use decoder
    elif conf['decoder']['status'] == True:
        encoder, _ = [l.lower() for l in conf['type'].split('/')]
        decoder_conf = conf['decoder']
        
        if decoder_conf['type'] == 'unet':
            architecture = smp.Unet(encoder,encoder_weights=None,in_channels=conf['in_channel'],classes=conf['out_channel'])

        elif decoder_conf['type'] == 'custom_unet': 
            architecture = custom_unet.CustomUnet(encoder,conf)

    

        else : 
            raise AttributeError(f'not support architecture config: {decoder_conf}')

    else:
        raise AttributeError(f'not support architecture config: {conf}')

    return architecture