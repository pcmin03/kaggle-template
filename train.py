import random , os 
import torch
import numpy as np
import hydra
import logging

from src import *
from typing import *
from torch.utils.data import DataLoader
from omegaconf import DictConfig

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
    os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"]= conf.base.env.gpus
    conf.optimizer.params.lr = conf.hyperparameter.lr
    conf.dataset.train.batch_size =   conf.dataset.test.batch_size = conf.hyperparameter.batch_size
    conf.dataset.valid.batch_size = conf.hyperparameter.batch_size
    if conf.scheduler.params.get('T_max', None) is None:
        conf.scheduler.params.T_max = conf.hyperparameter.epochs

@hydra.main(config_path="conf", config_name="cmpark_petfinder")
def run(conf:DictConfig) -> None:
    set_seed(conf)
    set_env(conf)
    model = architecture.create(conf.architecture) 
    optim = optimizer.create(conf.optimizer, model)
    sched = scheduler.create(conf.scheduler, optim)
    losses = loss.create(conf.loss)
    
    call_back  = Callback(conf)
    ema = call_back.load_ema(model)
    ealry_stop = call_back.load_early_stop()
    checkpoint_manger = call_back.load_checkpoint_manger()
    
    data_loader = {}
    for mode in ['train','valid']:
        datasetes = dataset.create(conf.dataset,mode=mode)
        data_loader[mode] = DataLoader(dataset=datasetes, 
                                    batch_size = conf.hyperparameter.batch_size,
                                    shuffle = True if mode == 'train' else False, 
                                    num_workers = conf.hyperparameter.num_workers, 
                                    # sampler = sampler if mode == 'train' else None,
                                    pin_memory = True)
    logger = log.create(conf)
    train = task.Petfinder_cls(conf,model=model,optim=optim,scheduler=sched,
                    log=logger,loss=losses,data=data_loader,saver=checkpoint_manger,
                    callback = {'ema':ema,'elary_stop':ealry_stop})

    train.run()
    train.test()

if __name__ == '__main__':
    run()
