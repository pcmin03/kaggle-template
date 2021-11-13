import os, shutil, random
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torch

from pathlib import Path 
from omegaconf import Container, OmegaConf
from typing import Dict,Optional
import logging  

from sklearn.metrics import confusion_matrix

from glob import glob
import itertools
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
LOGGER = logging.getLogger(__name__)

import mlflow
import mlflow.pytorch

from mlflow import log_matric,log_param, log_artifacts
import pickle

class Logger(object):
    ### save dictionary ###
    def __init__(self, 
                save_dir : str,
                version : str,
                add_parameter : bool, 
                log_graph : bool, 
                add_histogram : bool,
                conf) -> None :
        self.log_graph = log_graph
        self.add_parameter = add_parameter
        self.add_histogram = add_histogram 
        self.conf = conf

        # mlflow.set_traking_uri()
        base_path = os.path.join(save_dir,version) # set dir
        # check other board 
        
        current_v = glob(f'{base_path}/*')

        if not current_v:
            self.log_path = os.path.join(base_path,'v_0')
            if os.path.exists(self.log_path): 
                os.makedirs(self.log_path,exist_ok=True)

        else :
            current_v = list(map(str,current_v))
            version = list(map(lambda x: int(x.split('_')[-1]),current_v)) #check version list
            self.log_path = os.path.join(base_path,f'v_{max(version)+1}')
            if not os.path.exists(self.log_path): # makedir
                os.makedirs(self.log_path, exist_ok=True)
                # os.makedirs(self.log_path)
        print(self.log_path,'sdfasdfasdfadfasd')
        self.filename = self.make_filename(conf)
        self.writer = SummaryWriter(str(self.log_path),self.filename) # set logger path
        
    @property
    def return_path(self): 
        return self.log_path

    def make_filename(self,conf) -> str :
        return f'{conf.architecture.type}_{conf.optimizer.type}_{conf.loss.type}'

    def update_log(self,metrics:Dict[str,float],step: Optional[int],tag:str) -> None: 
        self.keepdic = {}
        for name,value in metrics.items(): 
            if isinstance(value,torch.Tensor): 
                value = value.item()

            if name == 'optimizer': 
                value = self.get_lr(value)
            
            if isinstance(value,dict): 
                self.writer.add_scalars(f'{tag}/{name}',value,step)
            else : 
                self.writer.add_scalar(f'{tag}/{name}',value,step)
            self.keepdic[f'{tag}/{name}'] = round(value,4)
        
    # def update_metric(self,predic:float,label:float,step: Optional[int] = None,tag:str) -> None:
        
    def update_histogram(self,parameters,step: Optional[int],tag:str) -> None:
        
        if self.add_histogram : 
            for name,params in parameters.state_dict().items():
                if isinstance(params,torch.Tensor): 
                    self.writer.add_histogram(f'{tag}/{name}',params,step)

    def update_graph(self,model,input_array:None) -> None:
        mlflow.pytorch.log_model(model, artifact_path="pytorch-model", pickle_module=pickle)

        if self.log_graph : 
            if input_array is None:    
                input_array = torch.rand((1,1,256,256)).float()
            
            self.writer.add_graph(model,input_array)

    def update_image(self,image,step: Optional[int],tag:str) -> None: 
        if isinstance(image,torch.Tensor): 
            image = image.detach().cpu().numpy()
            
        image=self.denormalize(image) # denormalize
        samplenum=random.randint(0,len(image)-1)
        
        # if image.ndim == 3: # batch 2d image
           
        #     image = image[samplenum][:,None]
        #     self.writer.add_images(f'{tag}/image',image[:10],step,dataformats='NCHW')
        #     self.writer.add_images(f'{tag}/image',image,step,dataformats='NCHW')

        if image.ndim == 4: # batch 2d image
           
            image = image[:20]
            self.writer.add_images(f'{tag}/image',image,step,dataformats='NCHW')
            # self.writer.add_images(f'{tag}/image',image,step,dataformats='NCHW')
        elif image.ndim == 5: # batch 3d image
            self.writer.add_images(f'{tag}/image',image[samplenum].transpose(1,0,2,3),step,dataformats='NCHW') # stacked 3d image
            print(image[samplenum:samplenum+1].transpose(0,2,1,3,4).shape,image[samplenum:samplenum+1].transpose(0,2,1,3,4).dtype)
            self.writer.add_video(f'{tag}/image',image[samplenum:samplenum+1].transpose(0,2,1,3,4),step) # plot 3d video 
# (N,3,H,W
#  (N, T, C, H, W)(N,T,C,H,W).
    def update_metric(self,label,predic,step: Optional[int],tag:str) -> None: 
        ax=self.confusionmetric(label,predic>0.6)
        rocurve = self.aurocplot(label,predic)
        self.writer.add_figure(f'{tag}/rocurve',rocurve,step) # plot confusion metric       
        self.writer.add_figure(f'{tag}/confusionmetric',ax,step) # plot confusion metric
        self.writer.add_pr_curve(f'{tag}',label,predic,step) # plot pr curve
        

    def update_parameter(self) -> None:

        if self.add_parameter:
            hyperparma = OmegaConf.to_container(self.conf.hyperparameter)
            self.writer.add_hparams(hyperparma,self.keepdic)

    #     if self.add_parameter: 
    #         self.writer.add_hparams(self.conf.hyperparameter,,run_name=self.filename)
            # self.writer.add_hparams(self.conf,run_name=self.filename)
            # self.conf.add_hparams
        
                
    def get_lr(self,optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def denormalize(self,image,max_value:int=255): # this is only gray scale image
        
        # if 
        # mean = np.array(self.conf.dataset.train.preprocess[-1].params.mean) * max_value
        # std = np.array(self.conf.dataset.train.preprocess[-1].params.std) * max_value
        
        # return ((image * std) + mean).astype(np.uint8) 
        return (image * max_value).astype(np.uint8)

    def check_onehotlist(self,onehostlist): 
        if isinstance(onehostlist,list): 
            onehostlist = np.array(onehostlist)

        if onehostlist.ndim > 1: 
            onehostlist = np.flatten(onehostlist)  # if array more than 1d, it make to 1d array
        
        return onehostlist
        
    def torch2numpy(slef,value): 
        if isinstance(value,torch.Tensor): 
            value = value.detach().cpu().numpy()
        return value

    def confusionmetric(self,label,pred): 
        
        import seaborn as sns # visulalization tool 

        categories = list(range(2)) # need to output channel 
        label_num = np.array(categories.copy())
        
        clean_metric = confusion_matrix(label,pred)
        
        # metric = clean_metric / label_num[:,None] * 100 # change 100 precentage
        ax = sns.heatmap(clean_metric, annot=True, fmt='d',cmap = 'YlGnBu',\
            xticklabels=categories,yticklabels=categories).get_figure()

        return ax
    def aurocplot(self,label,pred): 
        '''
        학습된 신경망과 배치로부터 가져온 이미지 / 라벨을 사용하여 matplotlib
        Figure를 생성합니다. 이는 신경망의 예측 결과 / 확률과 함께 정답을 보여주며,
        예측 결과가 맞았는지 여부에 따라 색을 다르게 표시합니다. "images_to_probs"
        함수를 사용합니다.
        '''
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        numclass = list(range(2))
        for i in numclass:
            fpr[i], tpr[i], _ = roc_curve(label,pred)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Compute micro-average ROC curve and ROC area
        fpr["micro"], tpr["micro"], _ = roc_curve(label, pred)
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # 배치에서 이미지를 가져와 예측 결과 / 정답과 함께 표시(plot)합니다
        fig,ax = plt.subplots(2,1,figsize=(10,10))
        for i in numclass:
            lw = 2
            ax[i].plot(fpr[i], tpr[i], color='darkorange',
                    lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[i])
            ax[i].plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
            ax[i].set_xlim([0.0, 1.0])
            ax[i].set_ylim([0.0, 1.05])
            ax[i].set_xlabel('False Positive Rate')
            ax[i].set_ylabel('True Positive Rate')
            ax[i].set_title('Receiver operating characteristic example')
            ax[i].legend(loc="lower right")

        return fig
        

