import torch
import logging

from .torch_ema import ModelEMA
from .pytorchtools import EarlyStopping
from .checkpoint import CheckpointManager

LOGGER = logging.getLogger(__name__)

class Callback(object): 
    def __init__(self,conf): 
        self.conf = conf

    def load_ema(self,model):
        conf = self.conf.callback.ema
        ema = ModelEMA(model,conf['decay'])
        return ema
        
    def load_early_stop(self):
        
        conf = self.conf.callback.early_stop        
        early_stop = EarlyStopping(mode=conf['mode'],patience=conf['patience'],path=conf['path'])
        # early_stop = EarlyStopping(mode=conf['mode'],1e-3,patience=conf['patience'],path=conf['path'])
        return early_stop
    
    def load_checkpoint_manger(self):
        
        conf = self.conf.callback.model_checkpoint
        saver = CheckpointManager('./',conf['mode'],conf['top_k'],conf['save_last'])
        return saver
