# -*- coding: utf-8 -*-
"""
Created on Sat Aug 14 15:35:44 2021

@author: baron015
"""

from torchvision import models
import torch.nn as nn
import torch.optim as optim
import torch


def get_model():
    model = models.segmentation.fcn_resnet50(True)
    model.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    model.aux_classifier[4] = nn.Conv2d(256, 1, kernel_size=(1, 1), stride=(1, 1))
    return model

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
    self.convRGB = nn.Sequential(nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
                                 nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                 nn.ReLU(inplace=True))
    self.convbands = nn.Sequential(nn.Conv2d(args.nb_channels-3, 64,kernel_size=(7,7), stride=(2,2), padding=(3, 3), bias=False),
                                   nn.BatchNorm2d(64,eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
                                   nn.ReLU(inplace=True))
    
    self.main = models.segmentation.fcn_resnet50(True)
    self.main.requires_grad  = False
    self.convRGB[0].weight.data = torch.clone(self.main.backbone.conv1.weight.data)
    self.convRGB[1].weight.data = torch.clone(self.main.backbone.bn1.weight.data)
    
    # emptying the first resnet conv
    ###### changer la resolution voir ce que ca fait
    self.main.backbone.conv1 = torch.nn.Identity()
    self.main.backbone.bn1 = torch.nn.Identity()
    self.main.backbone.relu = torch.nn.Identity()
    
    self.convRGB.requires_grad = False
    
        
    self.main.classifier[4] = nn.Conv2d(512, 1, kernel_size=(1, 1), stride=(1, 1))
    self.main.classifier[4].requires_grad  = True
    
    #weight initialization
    self.convbands[0].apply(self.init_weights)
    self.main.classifier[4].apply(self.init_weights)
    
    self.opti =  optim.Adam(self.parameters())
    
  def init_weights(self,layer): #gaussian init for the conv layers
  
    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
    
  def forward(self, input):
    """
    the function called to run inference
    after the model is created as an object 
    we call the method with the input as an argument
    """
    # passing through our layers
    x1bands = self.convbands(input[:,3:,:,:])
    x1RGB = self.convRGB(input[:,0:3,:,:])
    print(x1bands.shape)
    print(x1RGB.shape)
    
    x2 = self.main(x1bands+x1RGB)
    out = x2["out"]
    
    return out
          