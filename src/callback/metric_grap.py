from genericpath import exists
from typing import * 
import logging 
# Logger = logging.__name__ 
import os 
from glob import glob
import hydra
from pathlib import Path

class metric_gatter(object): 
    def __init__(self,conf,topk,mode): 
        self.conf = conf
        self.reset()
        self.model_save_name = conf.base.name
        self.checkpoint_path = Path(f'./{self.model_save_name}')
        # print(self.checkpoint_path.parents[0])
        self.checkpoint_path.mkdir(exist_ok=True)
        
        self.topk = topk 
        self.mode = mode

        self.valuesaver = {str(i):0 for i in range(topk)}

    def update(self,val:Dict[str,float]) -> None: 
        if isinstance(val,Dict): 
            if not self.collecter: 
                self._keeper(val)

            for k , v in val.items(): 
                self.collecter[k] = v
        
    def _keeper(self,val:Dict[str,float]) -> float:  
        for k,v in val.items(): 
            self.keepvalue[k] = v
        
    def checker(self,key:str): 
        for num,i in self.valuesaver.items(): 
            if i < self.collecter[key]: # k times save model
                self.valuesaver[num] =  self.collecter[key]
                return True
            elif isinstance(self.mode,str):
                if self.mode == 'max': 
                    result = self.keepvalue[key] < self.collecter[key]
                    if result:
                        self.valuesaver[num] = self.collecter[key]
                        self.keepvalue[key] = self.collecter[key]
                    return result

                elif self.mode == 'min': 
                    result = self.keepvalue[key] > self.collecter[key]
                    if result:
                        self.valuesaver[num] = self.collecter[key]
                        self.keepvalue[key] = self.collecter[key]
                    return result

            
    def checksave(self,key:str) -> bool:
        
        if self.keepvalue: 

            for num,i in self.valuesaver.items():

                if i == 0:
                    self.valuesaver[num] = self.collecter[key]
                    self.keepvalue[key] = self.valuesaver[num]
                    
                    return True

                else: 

                    if self.mode == 'max': 
                        result = self.valuesaver[num] > self.keepvalue[key]
                        self.valuesaver[num] = self.collecter[key]

                        if result:
                            self.keepvalue[key] = self.valuesaver[num]
                        return result
                            
                    elif self.mode == 'min': 
                        result = self.valuesaver[num] < self.keepvalue[key]
                        if result: 
                            self.keepvalue[key] = self.valuesaver[num]
                        return result
                

                
    def checkpointcheck(self,key:str,epoch) -> str: 
        
        
        listcheckpoint = list(self.checkpoint_path.glob('*.pth')) # previous checkpoint

        save_path_name = f'model-{epoch:03d}-{key}-{self.collecter[key]:.5f}.pth'  # new chechpoint
        
        if len(listcheckpoint) > self.topk :

            for num,i in enumerate(listcheckpoint): 
                
                if self.mode=='max' and self.collecter[key]  > float(str(i).split(key)[-1].replace('.pth','')):
                    listcheckpoint[num].unlink()
                    return self.checkpoint_path / save_path_name
                if self.mode=='min' and self.collecter[key]  < float(str(i).split(key)[-1].replace('.pth','')):
                    listcheckpoint[num].unlink()
                    return self.checkpoint_path / save_path_name
        elif len(listcheckpoint) <= self.topk: #             
            return self.checkpoint_path / save_path_name
                
    def reset(self):
        self.collecter = {}
        self.keepvalue = {}
    