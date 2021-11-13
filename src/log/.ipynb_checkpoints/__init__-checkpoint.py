import logging
from .logger import Logger

LOGGER = logging.getLogger(__name__)

def create(conf):
    if conf.log.type == 'tensorboard':
        savedir = conf.base.save_dir
        version = conf.log.version
        save_parameter = conf.log.save_parameter
        log_graph = conf.log.log_graph
        add_histogram = conf.log.add_histogram
        return Logger(savedir,version,save_parameter,log_graph,add_histogram,conf)
        
    else: 
        raise AttributeError(f'not support logger config: {conf}')

        