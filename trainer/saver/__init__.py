import torch
import logging

from . import checkpointsaver
from . import metric_grap

LOGGER = logging.getLogger(__name__)

def create(conf):
    # if conf['name'] == 'default_saver':
    saver = metric_grap.metric_gatter(conf,conf.saver.top_k,conf.saver.mode)
    
    # else:
    #     raise AttributeError(f'not support saver config: {conf}')

    return saver