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

    elif base == 'efficientv2':
        if architecture_name == 'S': 
            architecture= create_model("tf_efficientnetv2_s",in_chans=3, num_classes=1)
        elif architecture_name == 'M': 
            architecture = create_model("tf_efficientnetv2_m",in_chans=3, num_classes=1)
        # elif architecture_name == 'L': 
        architecture= create_model("tf_efficientnetv2_l",in_chans=40, num_classes=2)
        # architecture._conv_stem.in_chans = 40
        # print(architecture,architecture._conv_stem.weight.shape)
        # fistconv=nn.Conv2d(40,3,3,3)
        # architecture._conv_stem.weight = torch.nn.Parameter(torch.cat([architecture._conv_stem.weight, architecture._conv_stem.weight[:,0:1]], axis=1))
        # totalweigh = torch.cat([architecture._conv_stem.weight[:,0:1],torch.cat([architecture._conv_stem.weight]*13,axis=1)],axis=1)
        # architecture._conv_stem.weight = torch.nn.Parameter(totalweigh)
    # elif base == '3d':
    #     model = EfficientNet3D.from_name("efficientnet-b0", override_params={'num_classes': 2}, in_channels=4)
        
        # model.load_state_dict()
    elif base == 'efficientnet': 
        architecture = EfficientNet3D.from_name(f"{base}-{architecture_name}", override_params={'num_classes': 2}, in_channels=1)

    elif base: 
        architecture = create_model(base,in_chans=conf['in_channel'], num_classes=conf['out_channel'],pretrained=True)
    
    elif base == 'efficient':
        if architecture_name == 'b3': 
            architecture = EfficientNet.from_pretrained("efficientnet-b3", num_classes=1)
        elif architecture_name == 'b4': 
            architecture= EfficientNet.from_pretrained("efficientnet-b4", num_classes=1)
        elif architecture_name == 'b5': 
            architecture= EfficientNet.from_pretrained("efficientnet-b5", num_classes=1)
        elif architecture_name == 'b6': 
            architecture = EfficientNet.from_pretrained("efficientnet-b6", num_classes=1)
        # architecture._conv_stem = nn.Conv2d(40
        architecture._conv_stem.in_channels = 40
        # print(architecture,architecture._conv_stem.weight.shape)
        # fistconv=nn.Conv2d(40,3,3,3)
        # architecture._conv_stem.weight = torch.nn.Parameter(torch.cat([architecture._conv_stem.weight, architecture._conv_stem.weight[:,0:1]], axis=1))
        totalweigh = torch.cat([architecture._conv_stem.weight[:,0:1],torch.cat([architecture._conv_stem.weight]*13,axis=1)],axis=1)
        architecture._conv_stem.weight = torch.nn.Parameter(totalweigh)

        # print(architecture)
    elif base == 'resnet':
        
        if architecture_name == '34': 
            architecture = torch_models.resnet34(True,{num_classes:1})
        elif architecture_name == '50': 
            architecture = torch_models.resnet50(True,{num_classes:1})
        elif architecture_name == '101': 
            architecture = torch_models.resnet101(True,{num_classes:1})
        architecture.conv1 = nn.Conv2d(40, 64, kernel_size=7, stride=2, padding=3,
                            bias=False)
        architecture.fc = nn.Linear(2048,1)

    else:
        raise AttributeError(f'not support architecture config: {conf}')

    return architecture