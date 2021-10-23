import torch
import logging
import torchvision.models as torch_models
from torch import nn
from . import toy
LOGGER = logging.getLogger(__name__)
from efficientnet_pytorch import EfficientNet
# from .efficientv2 import EffNetV2
from timm.models import create_model
from efficientnet_pytorch_3d import EfficientNet3D

def create(conf, num_classes=None):
    base, architecture_name = [l.lower() for l in conf['type'].split('/')]
    print(conf,'51251215123124')
    if base == 'toy':
        architecture = toy.ToyModel()

    elif base: 
        architecture = create_model(base,in_chans=conf['in_channel'], num_classes=conf['out_channel'],pretrained=True)

    else:
        raise AttributeError(f'not support architecture config: {conf}')

    return architecture