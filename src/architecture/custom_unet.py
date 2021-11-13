import torch 
import torch.nn as nn 
import segmentation_models_pytorch as smp
        
class CustomUnet(nn.Module): 
    def __init__(self,encoder,conf): 
        super(CustomUnet, self).__init__()
        self.seg_model = smp.Unet(encoder,encoder_weights=None,in_channels=conf['in_channel'],classes=conf['out_channel'])
    
    def forward(self,x): 
        cls_predic = self.seg_model.encoder(x)
        seg_predic = self.seg_model(x)
        return seg_predic