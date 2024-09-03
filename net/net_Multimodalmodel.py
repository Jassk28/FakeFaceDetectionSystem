import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.config import cfg
from utils.basicblocks import BasicBlock
from utils.feature_fusion_block import DCT_Attention_Fusion_Conv
from utils.classifier import ClassifierModel

class Image_n_DCT(nn.Module):
    def __init__(self,):
        super(Image_n_DCT, self).__init__()
        self.Img_Block = nn.ModuleList()
        self.DCT_Block = nn.ModuleList()
        self.RGB_n_DCT_Fusion = nn.ModuleList()
        self.num_classes = len(cfg.CLASSES)



        for i in range(len(cfg.MULTIMODAL_FUSION.IMG_CHANNELS) - 1):
            self.Img_Block.append(BasicBlock(cfg.MULTIMODAL_FUSION.IMG_CHANNELS[i], cfg.MULTIMODAL_FUSION.IMG_CHANNELS[i+1], stride=1))
            self.DCT_Block.append(BasicBlock(cfg.MULTIMODAL_FUSION.DCT_CHANNELS[i], cfg.MULTIMODAL_FUSION.IMG_CHANNELS[i+1], stride=1))
            self.RGB_n_DCT_Fusion.append(DCT_Attention_Fusion_Conv(cfg.MULTIMODAL_FUSION.IMG_CHANNELS[i+1]))


        self.classifier = ClassifierModel(self.num_classes)
                


    def forward(self, rgb_image, dct_image):
        image = [rgb_image]
        dct_image = [dct_image]
        
        for i in range(len(self.Img_Block)):
            image.append(self.Img_Block[i](image[-1]))
            dct_image.append(self.DCT_Block[i](dct_image[-1]))
            image[-1] = self.RGB_n_DCT_Fusion[i](image[-1], dct_image[-1])
            dct_image[-1] = image[-1]
        out = self.classifier(image[-1])
        
        return out
    
