# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 15:35:44 2021

@author: baron015
"""

from torchvision.models.segmentation.deeplabv3 import DeepLabHead
from torchvision import models
import torch.nn as nn
import torch.optim as optim

# =============================================================================
# def createDeepLabv3(outputchannels=1):
#     """DeepLabv3 class with custom head
#     Args:
#         outputchannels (int, optional): The number of output channels
#         in your dataset masks. Defaults to 1.
#     Returns:
#         model: Returns the DeepLabv3 model with the ResNet101 backbone.
#     """
#     model = models.segmentation.deeplabv3_resnet101(pretrained=True,
#                                                     progress=True)
#     model.classifier = DeepLabHead(2048, outputchannels)
# 
#     return model
# =============================================================================


class TransferLearningModel(nn.Module):
  
  def __init__(self, args):
    """
    initialization function
    n_channels, int, number of input channel
    model_width, int list, size of the feature maps depth for the encoder after each conv
    decoder_conv_width, int list, size of the feature maps depth for the decoder after each conv
    n_class = int,  the number of classes
    """
    super(TransferLearningModel, self).__init__() #necessary for all classes extending the module class
    
    self = self.float()
    
    # creating first layer
    self.conv1 = nn.Conv2d(7,3,1)
    self.main = models.segmentation.deeplabv3_mobilenet_v3_large(pretrained=True)
    self.main.classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    
    # freezing the main part of the model
# =============================================================================
#     for param in next(self.children()).parameters():
#         param.requires_grad = False
# =============================================================================
    
    #weight initialization
    self.conv1.apply(self.init_weights)
    self.main.classifier[4].apply(self.init_weights)
    
# =============================================================================
#     self.opti =  optim.Adam([{'params': self.conv1.parameters()},
#                             {'params': self.main.classifier.parameters(),
#                             'lr': args.lr}], lr=args.lr/10, 
#                             weight_decay=0.001)
# =============================================================================
    self.opti = optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), lr=args.lr)
    
    
    
    
    
  def init_weights(self,layer): #gaussian init for the conv layers
  
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self, input):
    """
    the function called to run inference
    after the model is created as an object 
    we call the method with the input as an argument
    """
    
    # passing through our layers
    x1 = self.conv1(input)
    x2 = self.main(x1)
    out = x2["out"]
    
    return out
          