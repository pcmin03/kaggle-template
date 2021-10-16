import os
import sys
import logging
import datetime
import random
import numpy as np
import copy
import argparse
from contextlib import suppress

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn import DataParallel
import torch.nn.functional as F

import torchvision
import torchvision.transforms as transforms

import hydra
from omegaconf import DictConfig, OmegaConf

import trainer
# from trainer.utils import Callback_funcion

from tqdm import tqdm 
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score
import itertools
from torchmetrics import AUROC

import pandas as pd

class Trainer():
    def __init__(self, conf, rank=0):
        self.conf = copy.deepcopy(conf)
        self.rank = rank
        self.is_master = True if rank == 0 else False
        # self.is_master = False
        self.set_env()
        self.auroc = AUROC(pos_label=1)
        self.criterion_2 = nn.CrossEntropyLoss()
        self.sigmoid = nn.Sigmoid() 
        self.threhold = 0.5

    def set_env(self):
        torch.backends.cudnn.benchmark = True
        torch.cuda.set_device(self.rank)

        # mixed precision
        self.amp_autocast = suppress
        if self.conf.base.use_amp is True:
            self.amp_autocast = torch.cuda.amp.autocast
            self.scaler = torch.cuda.amp.GradScaler(enabled=True)
            
            if self.is_master:
                print(f'[Hyper]: Use Mixed precision - float16')
        else:
            self.scaler = None
        
        # Hyperparameter
        if self.is_master:
            print(f'[Hyper]: learning_rate: {self.conf.hyperparameter.lr} -> {self.conf.hyperparameter.lr * self.conf.base.world_size}')

        self.conf.optimizer.params.lr = self.conf.hyperparameter.lr * self.conf.base.world_size
        self.conf.dataset.train.batch_size =   self.conf.dataset.test.batch_size = self.conf.hyperparameter.batch_size
        self.conf.dataset.valid.batch_size = self.conf.hyperparameter.batch_size
        # Scheduler
        if self.conf.scheduler.type == 'CosineAnnealingLR': 
            if self.conf.scheduler.params.get('T_max', None) is None:
                self.conf.scheduler.params.T_max = self.conf.hyperparameter.epochs
        

        # chnage saver
        
        # warmup Scheduler
        # if self.conf.scheduler.warmup.get('status', False) is True:
        #     self.conf.scheduler.warmup.params.total_epoch = self.conf.hyperparameter.epochs

    def build_looger(self, is_use:bool):
        if is_use == True: 
            logger = trainer.log.create(self.conf)
            return logger
        else: 
            pass

    def build_model(self, num_classes=-1):
        model = trainer.architecture.create(self.conf.architecture)
        model = model.to(device=self.rank, non_blocking=True)
        model = DDP(model, device_ids=[self.rank], output_device=self.rank)
        return model

    def build_optimizer(self, model):
        optimizer = trainer.optimizer.create(self.conf.optimizer, model)
        return optimizer

    def build_scheduler(self, optimizer):
        scheduler = trainer.scheduler.create(self.conf.scheduler, optimizer)
        return scheduler

    # TODO: modulizaing
    def build_dataloader(self):
        dataloader = []
        for mode in ['train','valid','test']:
            dataloader.extend(trainer.dataset.create(
                self.conf.dataset,
                world_size=self.conf.base.world_size,
                local_rank=self.rank,
                mode=mode
            ))

        return dataloader

    def build_loss(self):
        criterion = trainer.loss.create(self.conf.loss, self.rank)
        criterion.to(device=self.rank, non_blocking=True)

        return criterion
    
    def build_saver(self, model, optimizer, scaler):
        saver = trainer.saver.create(self.conf.saver, model, optimizer, scaler)

        return saver


    def share_network(self,image,label,model,criterion):
        image = image.to(device=self.rank, non_blocking=True).float()
        label = label.to(device=self.rank, non_blocking=True).float()

        with self.amp_autocast():
            y_pred = model(image).squeeze()
            y_pred = self.sigmoid(y_pred)
            
            if self.conf.loss['type'] == 'MVL': 
                mean_loss, variance_loss = criterion(y_pred, label)
                loss = mean_loss + variance_loss
                loss += self.criterion_2(y_pred,label.long())

            elif self.conf.loss['type'] == 'MSE': 
                loss = criterion(y_pred, torch.stack((1-label,label),axis=1))
                
            elif self.conf.loss['type'] == 'ce':
                loss = criterion(y_pred, label.long())

            pos_pred = y_pred[:,1] # only need postivie label
            self.prediclist.append(pos_pred.detach().cpu().numpy())
            self.labellist.append(label.cpu().numpy())

        return loss,pos_pred.detach().cpu().numpy(),label.cpu().numpy()

    def cal_metric(self,loss,prediclist,labellist,logger,mode,optim=None): 
        # prediclist = np.clip(prediclist, 0, 1)
        # prediclist = np.nan_to_num(labellist)
        fpr,tpr,_  = metrics.roc_curve(labellist,prediclist,pos_label=1)
        prescore = precision_score(labellist,prediclist>0.5) #precision
        acccore = accuracy_score(labellist,prediclist>0.5) #acc
        # print(f'[Train_{epoch}] Acc: {train_hit / train_total} Loss: {one_epoch_loss / len(dl)}')
        metric = {'AUROC':metrics.auc(fpr, tpr),'Acc': acccore,'pre':prescore}
        
        if mode !='test':
            metric['Loss'] = loss
        if mode == 'train':
            metric['optimizer'] = optim
        
        logger.update_log(metric,self.current_step,mode) # update logger step
        logger.update_metric(labellist,prediclist,self.current_step,mode)
        return metric 

    def train_one_epoch(self, epoch, model, dl, criterion, optimizer,logger):
        
        model.train()
        pbar = self.setting_pbar(epoch,dl,'train')
        
        self.current_step = epoch
        self.prediclist = []
        self.labellist = []
        
        for step, (image, label) in pbar:
            # current_step = epoch*len(dl)+step
            
            loss,predic,label = self.share_network(image,label,model,criterion)
            optimizer.zero_grad(set_to_none=True)

            if self.scaler is None:
                loss.backward()
                optimizer.step()
                self.ema.update(model)
            else:
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()

            loss = loss.item()
            
            if step % 10 == 0:
                acc = accuracy_score(label,predic>self.threhold)
                pbar.set_postfix({'train_Acc':acc,'train_Loss':round(loss,2)}) 
        
        if self.is_master:
            labellist = np.array(list(itertools.chain(*self.labellist)))
            prediclist = np.array(list(itertools.chain(*self.prediclist)))
            
            _ = self.cal_metric(loss,prediclist,labellist,logger,'train',optimizer)
            logger.update_histogram(model,self.current_step,'train') # update weight histogram 
            logger.update_image(image,self.current_step,'train') # update transpose image
        
        return loss / len(dl), acc

    def setting_pbar(self,epoch,dl,mode):
        pbar = tqdm(
        enumerate(dl), 
        bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
        total=len(dl), 
        desc=f"{mode}:{epoch}/{self.conf.hyperparameter.epochs}"
        # disable=not self.is_master
        )
        return pbar

    @torch.no_grad()
    def eval(self, epoch, model, dl, criterion,logger):
        
        
        model.eval()
        pbar = self.setting_pbar(epoch,dl,'valid')
        current_step = epoch
        self.prediclist = []
        self.labellist = []

        for step, (image, label) in pbar:
            
            loss,predic,label = self.share_network(image,label,model,criterion)
            loss = loss.item()
            # if step % 1 == 0:
            acc = accuracy_score(label,predic>self.threhold)
            pbar.set_postfix({'valid_Acc':acc,'valid_Loss': round(loss, 2)}) 
            
        if self.is_master:
            labellist = np.array(list(itertools.chain(*self.labellist)))
            prediclist = np.array(list(itertools.chain(*self.prediclist)))
            self.valmetric = self.cal_metric(loss,prediclist,labellist,logger,'valid')
            logger.update_image(image,self.current_step,'valid') # update transpose image

        return loss / len(dl), acc

    def load_model(self, model, path):

        data = torch.load(path)
        key = 'model' if 'model' in data else 'state_dict'

        if not isinstance(model, (DataParallel, DDP)):
            model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})

        return model


    @torch.no_grad()
    def test(self,logger,model,test_dl):

        checkpoint_path = f'{self.conf.saver.checkpoint_save_path}/best_checkpoint.pth'
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = trainer.architecture.create(self.conf.architecture)
        model = model.to(device=device)
        checkpoint =torch.load(checkpoint_path)
        checkpoint = {key.replace("module.", ""): value for key, value in checkpoint.items()}
        model.load_state_dict(checkpoint)
        # model = self.load_model(model,checkpoint_path)
        # checkpoint = torch.load(checkpoint_path, map_location=device)
        
        

        # for key in list(checkpoint.keys()):
        #     if 'module.' in key:
                
        #         checkpoint[key.replace('module.', '')] = checkpoint[key]
        #         del checkpoint[key]#
        model.eval()
        predlist  = []
        idnames = []
        labellist = []
        tpbar = self.setting_pbar(0,test_dl,'test')

        for num,(image,label,pid) in tpbar:
            # current_step = epoch*len(dl)+step
            image = image.to(device=device, non_blocking=True).float()
            label = label.to(device=device, non_blocking=True).float()

            with self.amp_autocast():
                y_pred = model(image).squeeze()
                predlist.extend(self.sigmoid(y_pred)[:,1].flatten().tolist())
                labellist.extend(label.tolist())
                idnames.extend(pid)
                
        # if self.is_master:
        labellist = np.array(labellist)
        prediclist = np.array(predlist)
        pidlist = np.array(idnames)
        preddf = pd.DataFrame({"BraTS21ID": pidlist, "MGMT_value": prediclist}) 
        preddf = preddf.groupby('BraTS21ID',as_index=False).mean()
        
        labedf = pd.DataFrame({"BraTS21ID": pidlist, "MGMT_value": labellist}) 
        labedf = labedf.groupby('BraTS21ID',as_index=False).max()
        print(preddf)
        print(labedf)
        prediclist = np.array(preddf.MGMT_value)
        labellist = np.array(labedf.MGMT_value)
        logger.update_metric(labellist,prediclist,0,'test')
        # self.valmetric = self.cal_metric(None,prediclist,labellist,logger,'test')
        # logger.update_image(image,self.current_step,'test') # update transpose image

    def train_eval(self):
        model = self.build_model()
        criterion = self.build_loss()
        optimizer = self.build_optimizer(model)
        scheduler = self.build_scheduler(optimizer)
        train_dl, train_sampler,valid_dl,valid_sampler,self.test_dl,test_sampler = self.build_dataloader()
        logger = self.build_looger(is_use=self.is_master)
        callback= trainer.utils.Callback_funcion(self.conf)
        
        if self.is_master: 
            self.conf.saver.checkpoint_save_path = logger.log_path
            self.conf.saver.top_save_path = logger.log_path +'/top/'
            
            early_stopping = callback.load_early_stop(path=logger.log_path)
            self.ema = callback.load_ema(model)
            
        saver = self.build_saver(model, optimizer, self.scaler)
        # Wrap the model
        
        # initialize
        for name, x in model.state_dict().items():
            dist.broadcast(x, 0)
        torch.cuda.synchronize()

        # add graph to tensorboard
        if logger is not None:
            logger.update_graph(model,None)
        
        if self.conf.base.resume == True:
            self.start_epoch = saver.load_for_training(model,optimizer,self.rank,scaler=None)

        for epoch in range(1, self.conf.hyperparameter.epochs + 1):
            
            train_sampler.set_epoch(epoch)
            # train
            train_loss, train_acc = self.train_one_epoch(epoch, model, train_dl, criterion, optimizer, logger)
            
            
            # eval
            valid_loss, valid_acc = self.eval(epoch, model, valid_dl, criterion, logger)
            
            torch.cuda.synchronize()
            scheduler.step(valid_loss)
            # if self.is_master:
            print(f'Epoch {epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc:.3f}, train_Loss: {train_loss:.3f}, valid_Acc: {valid_acc:.3f}, valid_Loss: {valid_loss:.3f}')
            
            if self.is_master:
                saver.save_checkpoint(epoch=epoch, model=model, loss=valid_loss, rank=self.rank, metric=self.valmetric['AUROC'])
                early_stopping(self.valmetric['AUROC'], model,True)
                # early_stopping(self.valmetric['AUROC'])
                if early_stopping.early_stop:
                    print("Early stopping")
                    return logger,model,self.test_dl

        return logger,model,self.test_dl


    def run(self):
        if self.conf.base.mode == 'train':
            pass
        elif self.conf.base.mode == 'train_eval':
            logger,model,test_dl = self.train_eval()
            self.test(logger,model,test_dl)
        elif self.conf.base.mode == 'finetuning':
            pass
        

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

def runner(rank, conf):
    # Set Seed
    set_seed(conf)

    os.environ['MASTER_ADDR'] = str(conf.MASTER_ADDR)
    os.environ['MASTER_PORT'] = str(conf.MASTER_PORT)

    print(f'Starting train method on rank: {rank}')

    dist.init_process_group(
        backend='nccl', world_size=conf.base.world_size, init_method='env://',
        rank=rank
    )
    
    trainer = Trainer(conf, rank)

    trainer.run()
    

@hydra.main(config_path='conf', config_name='cmpark_fold1_single')
def main(conf: DictConfig) -> None:
    print(f'Configuration\n{OmegaConf.to_yaml(conf)}')
    
    mp.spawn(runner, nprocs=conf.base.world_size, args=(conf, ))
    

if __name__ == '__main__':
    main()
