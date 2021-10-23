import random , os , itertools, yaml,argparse
from pandas.io.sql import read_sql_query
import numpy as np
from glob import glob
from natsort import natsorted

# torch code
import pytorch_lightning as pl
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torchsummary

from pytorch_lightning.callbacks import EarlyStopping,ModelCheckpoint,LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import ExponentialLR
from torchmetrics.functional import accuracy

# my code
from HER2dataloader import HER2dataloderpl , inferHER2dataloderpl
from utils import *
from models import choosemodel
from pytorch_grad_cam import GradCAM,ScoreCAM
import cv2

from custom_loss import * 
from torch.nn import functional as F

import hydra
from omegaconf import DictConfig, OmegaConf
import logging

from metric_grap import metric_gatter
import os
from path import Path
import torch 
import trainer

# from 


LOGGER = logging.getLogger(__name__)
LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv('RANK', -1))
WORLD_SIZE = int(os.getenv('WORLD_SIZE', 1))

def set_seed(conf):
    if conf.base.seed is not None:
        conf.base.seed = int(conf.base.seed, 0)
        print(f'[Seed] :{conf.base.seed}')
        os.environ['PYTHONHASHSEED'] = str(conf.base.seed)
        random.seed(conf.base.seed)
        np.random.seed(conf.base.seed)
        torch.manual_seed(conf.base.seed)
        torch.cuda.manual_seed(conf.base.seed)
        torch.cuda.manual_seed_all(conf.base.seed)  # if use multi-GPU
        torch.backends.cudnn.deterministic = True

log = logging.getLogger(__name__)
@hydra.main(config_path="conf", config_name="None")
def run(conf:DictConfig) -> None:


    

    
    logger = trainer.log.create(conf)
    # Scheduler
    # if conf.scheduler.params.get('T_max', None) is None:
    #    conf.scheduler.params.T_max = conf.hyperparameter.epochs


    model = trainer.architecture.create(conf.architecture)
    optimizer = trainer.optimizer.create(conf.optimizer, model)
    scheduler = trainer.scheduler.create(conf.scheduler, optimizer)

    
    conf.optimizer.params.lr = conf.hyperparameter.lr * conf.base.world_size
    conf.dataset.train.batch_size =   conf.dataset.test.batch_size = conf.hyperparameter.batch_size
    conf.dataset.valid.batch_size = conf.hyperparameter.batch_size
    conf.saver.checkpoint_save_path = logger.log_path
    conf.saver.top_save_path = logger.log_path +'/top/'

    device = trainer.utils.torch_tuils.select_device(conf.base.gpus,conf.hyperparameter.batch_size)
    
    early_stop =EarlyStopping(patience=40, verbose=True,path=logger.log_path)

    torch.cuda.set_device(LOCAL_RANK)
    device = torch.device('cuda', LOCAL_RANK)

    dist.init_process_group(
        backend="nccl" if dist.is_nccl_available() else "gloo", 
        world_size=conf.base.world_size
        # init_method='env://',rank=rank
    )

    saver = trainer.saver.create(self.conf.saver, model, optimizer, scaler)

    # for i in 
    # train_loader, train_sampler = trainer.dataset.create(
    #         self.conf.dataset,
    #         world_size=self.conf.base.world_size,
    #         local_rank=self.rank,
    #         mode='train'
    #     )

    
    train = Trainer(conf,model=model,optim=optimizer,scheduler=scheduler,\
                        ealry_stop=early_stop,device=device,
                        log=logger)

    train.run()
    
    # if WORLD_SIZE > 1 and RANK == 0:
    #     LOGGER.info('Destroying process group... ')
    #     dist.destroy_process_group()

    # def build_dataloader(self, ):
        
    #     dataloader_dic = {}
    #     for mode in ['train','valid','test']: 
    #         dataloader_dic[mode] = trainer.dataset.create(
    #             self.conf.dataset,
    #             world_size=self.conf.base.world_size,
    #             local_rank=self.rank,
    #             mode=mode
    #         )

    #     return dataloader_dic

    # def build_loss(self):
    #     criterion = trainer.loss.create(self.conf.loss, self.rank)
    #     criterion.to(device=self.rank, non_blocking=True)

    #     return criterion
    
    # def build_saver(self, model, optimizer, scaler):
    #     saver = trainer.saver.create(self.conf.saver, model, optimizer, scaler)
    #     return saver

    # log.info("Start training model")
    # log.debug("Debug level message")    
    # print(f'Workdir : {os.getcwd()}')

    # os.environ['CUDA_VISIBLE_DEVICES'] = str(conf.base.opt_gpus)
    # set_seed(conf.base.randomseed)

    # # change main path 
    # conf.base.checkpoint_path = f'{hydra.utils.get_original_cwd()}/{conf.base.checkpoint_path}'
    # # conf.base.tensorboard_path = f'{hydra.utils.get_original_cwd()}/{conf.base.tensorboard_path}'
    # print(conf.base.checkpoint_path,conf.base.tensorboard_path)

    # # set a callback function 
    # checkpoint_callback = ModelCheckpoint(
    #                     monitor='valid_loss',
    #                     dirpath=conf.base.checkpoint_path,
    #                     filename=conf.base.checkpoint_path,
    #                     save_top_k=3,
    #                     mode='min')

    # ealrystopping = EarlyStopping(monitor='valid_sens/score2+',
    #                             min_delta=0.003, 
    #                             patience = 20, 
    #                             verbose=False,
    #                             mode='max')
    # # load learnigmonitor
    # lr_monitor = LearningRateMonitor(logging_interval='step')

    # # set tensorboard 
    # tb_logger = Tensorboard(conf.base.checkpoint_path,name=conf.base.tensorboard_path)
    
    # # set a trainer 
    # trainer = pl.Trainer(
    #                     max_epochs=conf.hyperparameter.epochs, 
    #                     gpus = conf.base.gpus,
    #                     default_root_dir=conf.base.checkpoint_path,
    #                     logger = tb_logger,
    #                     check_val_every_n_epoch=1,
    #                     callbacks=[lr_monitor])
    
    # # select model 
    # mymodel = choosemodel(conf.model)

    # # load loss
    # conf.loss.param.end_age = conf.model.out_ch

    # loss = MeanVarianceLoss(conf.loss) #noise robust loss
    # # dataset module
    # if conf.base.train == True:
    #     conf.dataset.train.batch_size = conf.hyperparameter.batch_size
    #     her2data = HER2dataloderpl(conf.dataset.train)
    #     model = HER2classify(mymodel,loss,conf)
        
    #     # torchsummary.summary(model,(1,3,256,256),device='cpu')
    #     # torchsummary()
    #     trainer.fit(model,her2data)
    
    # dataset inferencedataload
#     else: 
#         her2data = inferHER2dataloderpl(opt['base_path'], opt['corr_path'],opt['wsin'],opt['roin'],opt['batchsize'],opt)
#         checkpath = opt['checkpoint_path']
#         last_savename = natsorted(glob(f'{checkpath}/*.ckpt'))[-1] # import last checkpoint
#         #torch version
#         model_state_dict = torch.load(last_savename)

    
#         her2data.setup(stage='test',useopenslide=opt['useopenslide'])
#         from torch.utils.data import DataLoader
#         import torch 
        
#         dataloder = DataLoader(her2data.her2test,shuffle=False,num_workers=8,
#                           batch_size=her2data.batch_size)
        
#         # prediction
#         sample = model_state_dict['state_dict']
#         print(sample.keys())
#         for key in list(sample.keys()):
#             sample[key.replace('encoder.', '')] = sample.pop(key)
#         mymodel.load_state_dict(sample, strict=True)
        
#         mymodel.eval()
#         mymodel.cuda().float()
#         from tqdm import tqdm
#         score = []
#         resultimg = []
#         for i in tqdm(dataloder): 
#             with torch.no_grad():
#                 predics = torch.argmax(F.softmax(mymodel(i[0].cuda()),dim=1),dim=1).detach().cpu().numpy()
#                 print(predics)
#                 score.append(predics)
#                 resultimg.append(i[1])
#         # ligthning version
# #         model = HER2classify.load_from_checkpoint(checkpoint_path=last_savename,model=mymodel,lr=opt['lr'],opt=opt)
# #         print(model.state_dict())
# #         her2data.setup(stage='test',useopenslide=opt['useopenslide'])
        
# #         trainer.test(model,datamodule=her2data)
        
#         if opt['WSIsave'] == True: 
# #             result = np.stack(np.array(model.val_collector),axis=1)        
# #             score =list(itertools.chain(*result[0]))
#             score =list(itertools.chain(*score))
#             print(score)
# #             resultimg =list(itertools.chain(*result[1]))
# #             resultimg =list(itertools.chain(*resultimg))

#             # save result score & path
#             # basepathes = opt['base_path']
#             # savecsv([resultpath,score],f'{basepathes}')

#             if opt['useopenslide'] == True:
#                 location = her2data.roipatch[:,1]
#                 patchimages = her2data.images
#                 slidename = her2data.slidename
#                 print(location,np.stack(location),'123123123123')
#                 recover_wsi(patchimages,np.stack(location),score,opt['checkpoint_path'],slidename)

#             elif opt['useopenslide'] == False:
#                 make_wsi(resultimg,opt['corr_path'],opt['wsin'],opt['roin'],score,opt['checkpoint_path'],slidename)

if __name__ == '__main__':
    ### config list
    
    run()
