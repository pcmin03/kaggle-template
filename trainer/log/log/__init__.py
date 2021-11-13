import logging
from .logger import Logger
# from .mlflow import ML_logger

LOGGER = logging.getLogger(__name__)

def create(conf):
    if conf.log.type == 'tensorboard':
        
        version = conf.log.version
        save_parameter = conf.log.save_parameter
        log_graph = conf.log.log_graph
        add_histogram = conf.log.add_histogram
        return Logger(version,save_parameter,log_graph,add_histogram,conf)
    # elif conf.log.type == 'mlflow': 
        
    #     return ML_logger

    else: 
        raise AttributeError(f'not support logger config: {conf}')

        