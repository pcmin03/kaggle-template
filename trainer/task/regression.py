import numpy as np
import copy
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

# from trainer.utils import Callback_funcion

from tqdm import tqdm 
from sklearn.metrics import roc_auc_score
from sklearn import metrics
from sklearn.metrics import precision_score, accuracy_score

import pandas as pd
from accelerate import Accelerator
from collections import defaultdict

class Regression:
    def __init__(self, conf, model,optim,scheduler,
                    log,loss,data,saver):
        self.conf = copy.deepcopy(conf)

        # self.is_master = False
        self.accelerator = Accelerator(
            device_placement = True,
            split_batches = False,
            fp16 = self.conf.base.use_amp == True,
            cpu = False,
            deepspeed_plugin = None,
            rng_types = None,
            kwargs_handlers = None
        )

        self.device    = self.accelerator.device
        self.model,self.optim = self.accelerator.prepare(model,optim)
        self.train_dl,self.valid_dl = self.accelerator.prepare(data['train'],data['valid'])
        self.criterion  = self.accelerator.prepare(loss)
        self.sched     = scheduler
        self.saver     = saver
        self.log       = log
        self.saver     = saver
        self.threhold = 0.5
        self.disable_tqdm = not self.accelerator.is_local_main_process
        self.sigmoid = nn.Sigmoid()

    @property
    def current_epoch(self):
        return self._current_epoch    

    @property 
    def current_stop(self):
        return self._current_step

    def setting_pbar(self,epoch,dl,mode):
        pbar = tqdm(
                    enumerate(dl), 
                    bar_format='{desc:<15}{percentage:3.0f}%|{bar:18}{r_bar}', 
                    total=len(dl), 
                    desc=f"{mode}:{epoch}/{self.conf.hyperparameter.epochs}",
                    disable=self.disable_tqdm)
        return pbar

    def share_network(self,image,label,model,pbar):
        image = image.to(self.device, non_blocking=True).float()
        label = label.to(self.device, non_blocking=True).float()

        y_pred = model(image).squeeze()
        y_pred = self.sigmoid(y_pred)
        
        if self.conf.loss['type'] == 'MVL': 
            mean_loss, variance_loss = self.criterion(y_pred, label)
            loss = mean_loss + variance_loss
            loss += self.criterion_2(y_pred,label.long())

        elif self.conf.loss['type'] == 'MSE': 
            loss = self.criterion(y_pred, torch.stack((1-label,label),axis=1))
            
        elif self.conf.loss['type'] == 'ce':
            loss = self.criterion(y_pred, label.long())
            y_pred = torch.argmax(y_pred,axis=1)
        else : 
            assert "no loss define"

        all_predictions = self.accelerator.gather(y_pred).detach().cpu().numpy()
        all_labels = self.accelerator.gather(label).detach().cpu().numpy()
        
        # self.train_dic['predic'].append(all_predictions.detach())
        # self.train_dic['labels'].append(all_labels.detach())
        
        if self.current_epoch % 10 == 0:
            # fpr,tpr,_  = metrics.roc_curve(all_labels,all_predictions,pos_label=1)
            pbar.set_postfix({'train_Loss':round(loss.item(),2)}) 

        return loss.item(),all_predictions,all_labels

    def cal_metric(self,loss,prediclist,labellist,mode): 
        
        if not isinstance(prediclist,np.ndarray): 
            
            prediclist = np.array(prediclist)
            labellist = np.array(labellist)

        # fpr,tpr,_  = metrics.roc_curve(labellist,prediclist,pos_label=1)
        prescore = precision_score(labellist,prediclist,pos_label=1) #precision
        acccore = accuracy_score(labellist,prediclist,pos_label=1) #acc
        # print(f'[Train_{epoch}] Acc: {train_hit / train_total} Loss: {one_epoch_loss / len(dl)}')
        metric = {'Acc': acccore,'pre':prescore}
        
        if mode !='test':
            metric['Loss'] = loss
        if mode == 'train':
            metric['optimizer'] = self.optim
        
        self.log.update_log(metric,self.current_epoch,mode) # update self.log step
        self.log.update_metric(labellist,prediclist,self.current_epoch,mode)
        return metric 

    def train_one_epoch(self, model, dl, criterion):
        
        model.train()
        
        pbar = self.setting_pbar(self.current_epoch,dl,'train')
        train_dic = defaultdict(list)
        for step, (image, label) in pbar:
            self._current_step = self.current_epoch*len(dl)+step
            
            loss,predic,label = self.share_network(image,label,model,criterion,pbar)
            train_dic['predic'].extend(predic)    
            train_dic['label'].extend(label)    

            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.optim.stop()
        
        train_matric = self.cal_metric(loss,train_dic['predic'],train_dic['label'],'train',self.optim)
        self.log.update_histogram(model,self.current_epoch,'train') # update weight histogram 
        self.log.update_image(image,self.current_epoch,'train') # update transpose image
        
        return loss / len(dl), train_matric


    @torch.no_grad()
    def eval(self, model, dl,testing=False):
        
        model.eval()
        pbar = self.setting_pbar(self.current_epoch,dl,'valid')
        valid_dic = defaultdict(list)
        for step, (image, label) in pbar:
            
            loss,predic,label = self.share_network(image,label,model,pbar)
            valid_dic['predic'].extend(predic)    
            valid_dic['label'].extend(label)    
            
        
        valmetric = self.cal_metric(loss,valid_dic['predic'],valid_dic['label'],'valid')
        self.log.update_image(image,self.current_epoch,'valid') # update transpose image
        if testing == True: 
            return 0
            
        return loss / len(dl), valmetric

    @torch.no_grad()
    def test(self, model, dl,testing=False):
        
        model.eval()
        pbar = self.setting_pbar(self.current_epoch,dl,'valid')
        test_dic = defaultdict(list)
        for step, (image, label) in pbar:
            
            loss,predic,label = self.share_network(image,label,model,pbar)
            test_dic['predic'].extend(predic)    
            test_dic['label'].extend(label)    

        valmetric = self.cal_metric(loss,test_dic['predic'],test_dic['label'],self.log,'valid')
        self.log.update_image(image,self.current_epoch,'valid') # update transpose image
        if testing == True: 
            return 0
            

        return loss / len(dl), valmetric
    def load_model(self, model, path):

        data = torch.load(path)
        key = 'model' if 'model' in data else 'state_dict'

        if not isinstance(model, (DataParallel, DDP)):
            model.load_state_dict({k.replace('module.', ''): v for k, v in data[key].items()})
        else:
            model.load_state_dict({k if 'module.' in k else 'module.'+k: v for k, v in data[key].items()})

        return model

    def run(self):
                
        # add graph to tensorboard
        if self.log is not None:
            self.log.update_graph(self.model,None)
        
        for epoch in range(1, self.conf.hyperparameter.epochs + 1):
            self._current_epoch = epoch
            self.prediclist = []
            self.labellist = []
            #just testing 
            _ = self.eval(self.model, self.valid_dl,True)
            # train
            train_loss, train_acc = self.train_one_epoch(self.model, self.train_dl)
            # eval
            valid_loss, valid_acc = self.eval(self.model, self.valid_dl,False)
            
            print(train_acc)
            self.scheduler.step(self.current_epoch)
            # if self.is_master:
            print(f'Epoch {self.current_epoch}/{self.conf.hyperparameter.epochs} - train_Acc: {train_acc:.3f}, train_Loss: {train_loss:.3f}, valid_Acc: {valid_acc:.3f}, valid_Loss: {valid_loss:.3f}')
            self.disable_tqdm
            self.accelerator.wait_for_everyone()
    
    def test(self): 
        test_loss, test_acc = self.test(self.model, self.valid_dl,False)


            # if self.is_master:
            #     saver.save_checkpoint(epoch=epoch, model=model, loss=valid_loss, rank=self.rank, metric=self.valmetric['AUROC'])
            #     early_stopping(self.valmetric['AUROC'], model,True)
            #     # early_stopping(self.valmetric['AUROC'])
            #     if early_stopping.early_stop:
            #         print("Early stopping")
            #         return self.log,model,self.test_dl