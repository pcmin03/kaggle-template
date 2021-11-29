import random , os 
import torch
import numpy as np

import trainer
from trainer import log,architecture,optimizer,scheduler,utils,dataset,saver,task
# from trainer.saver  import metric_grap
from trainer.utils import Callback_funcion
from omegaconf import DictConfig

from typing import *
from torch.utils.data import DataLoader, WeightedRandomSampler
import hydra
import logging

def set_seed(conf : DictConfig) -> None:
    conf.base.seed = int(conf.base.seed, 0)
    print(f'[Seed] :{conf.base.seed}')
    os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
    random.seed(conf.base.seed)
    np.random.seed(conf.base.seed)
    torch.manual_seed(conf.base.seed)
    torch.cuda.manual_seed(conf.base.seed)
    torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True

def set_env(conf : DictConfig) -> None: 

    conf.optimizer.params.lr = conf.hyperparameter.lr
    conf.dataset.train.batch_size =   conf.dataset.test.batch_size = conf.hyperparameter.batch_size
    conf.dataset.valid.batch_size = conf.hyperparameter.batch_size
    if conf.scheduler.params.get('T_max', None) is None:
        conf.scheduler.params.T_max = conf.hyperparameter.epochs

    # conf.saver.checkpoint_save_path = conf.logger.log_path
    # conf.saver.top_save_path = conf.logger.log_path +'/top/'

# log = logging.getLogger(__name__)
@hydra.main(config_path="conf", config_name="cmpark")
def run(conf:DictConfig) -> None:
    
    set_env(conf)
    logger    = log.create(conf)
    model     = architecture.create(conf.architecture)
    optim = optimizer.create(conf.optimizer, model)
    sched = scheduler.create(conf.scheduler, optim)
    losses    = trainer.loss.create(conf.loss)
    checkpoint_manger     = saver.create(conf)
    # callback  = Callback_funcion(conf)

    # ema = callback.load_ema(model)
    # ealry_stop = callback.ealry_stop(patience=40, verbose=True,path=logger.log_path)
    
    # saver = trainer.saver.create(self.conf.saver, model, optimizer)
    ## setting gpu device
    
    
    # saver = CheckpointManager(
    #         save_root=conf.saver.top_save_path,
    #         mode=conf.saver. ,
    #         top_k=conf.saver.top_save_path
    #     )

    data_loader = {}
    for mode in ['train','valid','test']:
        datasetes = dataset.create(conf.dataset,mode=mode)
        data_loader[mode] = DataLoader(dataset=datasetes, 
                                    batch_size = conf.hyperparameter.batch_size,
                                    shuffle = True if not mode == 'test' else False, 
                                    num_workers = conf.hyperparameter.num_workers, 
                                    # sampler = sampler if mode == 'train' else None,
                                    pin_memory = True)

    train = task.Classify(conf,model=model,optim=optim,scheduler=sched,
                    log=logger,loss=losses,data=data_loader,saver=checkpoint_manger)

    train.run()
    train.test()

if __name__ == '__main__':
    run()
