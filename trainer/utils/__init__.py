import torch
import logging

from .torch_ema import ModelEMA
from .pytorchtools import EarlyStopping

LOGGER = logging.getLogger(__name__)

class Callback_funcion(object): 
    def __init__(self,conf): 
        self.conf = conf

    def load_ema(self,model):
        conf = self.conf.utils.ema
        ema = ModelEMA(model,conf['decay'])
        return ema
        
    def load_early_stop(self,**kwargs):
        
        conf = self.conf.utils.early_stop
        param = list(conf.keys())
        
        for k,v in kwargs.items():
            if k in param:
                conf[k] = v
        
        early_stop = EarlyStopping(patience=conf['patience'], verbose=conf['verbos'],path=conf['path'])
        # early_stop = EarlyStopping(mode=conf['mode'],1e-3,patience=conf['patience'],path=conf['path'])
        return early_stop

