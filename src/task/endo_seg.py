import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm 
from accelerate import Accelerator
from ..metric import MetricTracker

class Endoseg:
    def __init__(self, conf, model,optim,scheduler,
                    log,loss,data,saver,callback):
        self.conf = copy.deepcopy(conf)

        # torch.cuda.device_count() > 1
        self.accelerator = Accelerator(
            device_placement = True,
            split_batches = False,
            fp16 = self.conf.base.use_amp == True,
            cpu = False,
            deepspeed_plugin = None,
            rng_types = None,
            kwargs_handlers = None
        )
        
        self.model,self.optim = self.accelerator.prepare(model,optim)
        self.train_dl,self.valid_dl = self.accelerator.prepare(data['train'],data['valid'])
        self.criterion = self.accelerator.prepare(loss)
        self.device    = self.accelerator.device
        self.sched     = scheduler
        self.saver     = saver
        self.log       = log
        self.disable_tqdm = not self.accelerator.is_local_main_process
        self.sigmoid = nn.Softmax(dim=1)
        self.train_metric  = MetricTracker(2)
        self.valid_metric  = MetricTracker(2)
        self.callback = callback
        self.avg = 'macro'
        self.target_metric = 'recall'
        # self.criterion_2 = nn.CrossEntropyLoss(self.conf.loss['ignore_index'])

    @property
    def current_epoch(self):
        return self._current_epoch    

    @property 
    def current_step(self):
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
        label = label.to(self.device, non_blocking=True).long()
        
        predic = model(image)
        predic = self.sigmoid(predic)

        if self.conf.loss['type'] == 'MVL': 
            print(predic.shape)
            mean_loss, variance_loss = self.criterion(predic.view, label)
            loss = mean_loss + variance_loss
            loss += self.criterion_2(predic,label)
        
        elif self.conf.loss['type'] == 'MSE': 
            loss = self.criterion(predic, F.one_hot(label,3).permute(0,3,1,2).float())
        elif self.conf.loss['type'] == 'ce':
            loss = self.criterion(predic,label)
            # predic = torch.argmax(predic,axis=1)    
        else :
            assert "no loss define"

        predic = self.accelerator.gather(predic).detach().cpu().numpy()
        label = self.accelerator.gather(label).detach().cpu().numpy()

        # if self.current_epoch % 10 == 0:
        pbar.set_postfix({'Loss':np.round(loss.item(),2)}) 

        return loss,predic,label
    
    def _train_one_epoch(self, model, dl):
        
        model.train()
        self.train_metric.reset()
        pbar = self.setting_pbar(self.current_epoch,dl,'train')
        
        for step, (image, label) in pbar:
            self._current_step = self.current_epoch*len(dl)+step

            loss,predic,label = self.share_network(image,label,model,pbar)
            self.train_metric.update(label,predic)

            self.optim.zero_grad()
            self.accelerator.backward(loss)
            self.optim.step()
            # matric = self.train_metric.seg_metric(self.avg)

        if self.current_epoch % self.conf.base.log_epoch == 0:
            self.train_metric.output_dic['predic'] = np.max(np.array(self.train_metric.output_dic['predic']),axis=1)
            matric = self.train_metric.seg_metric(self.avg,0.5)
            print(f'train:{matric}') 

            matric.update({'optimizer':self.optim})
            self.log.update_log(matric,self.current_epoch,'train') # update self.log step
            self.log.update_image(image[0],self.current_epoch,'train','img') # update transpose image
            self.log.update_image(label[0],self.current_epoch,'train','label') # update transpose image
            self.log.update_image(predic[0,1],self.current_epoch,'train','predic_pos') # update transpose image
        
        else : 
            matric = None

        return loss,matric


    @torch.no_grad()
    def _valid_one_epoch(self, model, dl):
        
        model.eval()
        pbar = self.setting_pbar(self.current_epoch,dl,'valid')
        self.valid_metric.reset()
        for step, (image, label) in pbar:
            loss,predic,label = self.share_network(image,label,model,pbar)   
            self.valid_metric.update(label,predic)
        
        self.valid_metric.output_dic['predic'] = np.max(np.array(self.valid_metric.output_dic['predic']),axis=1)
        matric = self.valid_metric.seg_metric(self.avg,0.5) 
        print(f'valid:{matric}')
        
        self.log.update_log(matric,self.current_epoch,'valid') # update 
        self.log.update_image(image[0],self.current_epoch,'valid','img') # update transpose image
        self.log.update_image(label[0],self.current_epoch,'valid','label') # update transpose image
        self.log.update_image(predic[0,1],self.current_epoch,'valid','predic_pos') # update transpose image
        
        return loss,matric

    def run(self):
        # add graph to tensorboard
        if self.log is not None:
            self.log.update_graph(self.model,None)
        
        for epoch in range(1, self.conf.hyperparameter.epochs + 1):
            self._current_epoch = epoch
            # train
            train_loss,train_metric = self._train_one_epoch(self.model, self.train_dl)
            
            if self.sched != None: 
                self.sched.step(self.current_epoch)
            # eval
            if self.current_epoch % self.conf.base.val_check == 0:

                valid_loss,valid_metric = self._valid_one_epoch(self.model, self.valid_dl)

                self.accelerator.wait_for_everyone()
                sd = {'checkpoint' : self.accelerator.unwrap_model(self.model).state_dict(),
                    'metric':valid_metric}

                self.saver.step(valid_metric[self.target_metric],sd,epoch)
                self.callback['elary_stop'](valid_metric[self.target_metric],sd)
            
            print(f'Epoch {self.current_epoch}/{self.conf.hyperparameter.epochs} , train_Loss: {train_loss:.3f}, valid_Loss: {valid_loss:.3f}')
        