import torch
import logging

from . import checkpointsaver

LOGGER = logging.getLogger(__name__)

def create(conf, model, optimizer, scaler):
    if conf['name'] == 'default_saver':
        saver = checkpointsaver.CheckpointSaver(conf, model, optimizer, scaler=scaler)
    else:
        raise AttributeError(f'not support saver config: {conf}')

    return saver