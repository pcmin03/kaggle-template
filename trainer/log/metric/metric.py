import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import precision_score,recall_score,f1_score,accuracy_score,roc_curve,auc
warnings.filterwarnings("ignore")
from collections import defaultdict
from sklearn.metrics import PrecisionRecallDisplay

# Loss Meter
class MetricTracker:
    def __init__(self,classnum:int) -> None : 
        self.classnum = classnum
        self.reset()

    def cal_iou(self,label,predic):
        intersection = np.logical_and(label, predic)
        union = np.logical_or(label, predic)
        iou_score = np.sum(intersection) / np.sum(union)
        return iou_score

    def update(self,label,predic): 
        self.output_dic['predic'].extend(predic)
        self.output_dic['label'].extend(label)
    
    def seg_metric(self,avg,threhold): 
        predic = self.output_dic['predic']
        label  = self.output_dic['label']
        label = np.array(label).flatten()

        if not isinstance(predic,np.ndarray): 
            predic = np.array(predic).flatten()
        else: 
            predic = predic.flatten()

        
        fpr,tpr,_  = roc_curve(label,predic,pos_label=2)
        roc_auc = auc(fpr, tpr)
        print(predic.shape,label.shape,predic)

        spec = precision_score(label,predic>threhold,average=avg) # precision
        sens = recall_score(label,predic>threhold,average=avg) # recall
        f1 = f1_score(label,predic>threhold,average=avg) # recall
        acc = accuracy_score(label,predic>threhold) # acc

        metric = {'Acc': acc,'pre':spec,'recall':sens,'f1':f1,'roc':roc_auc}
        
        
        return metric 

    def reset(self): 
        self.output_dic = defaultdict(list)
