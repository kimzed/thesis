# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:01:31 2021

@author: baron015
"""


import torch.nn as nn
import torch
import torch.optim as optim



class CnnSemanticSegmentation(nn.Module):
  
  def __init__(self):
    super(CnnSemanticSegmentation, self).__init__()
    
    self = self.float()
    if torch.cuda.is_available():
        self.cuda()
    self.learning_rate = 0.02

    self.number_hidden_channels = 32
    self.number_input_channels = 6

    self.Convolution1 = nn.Sequential(nn.Conv2d(self.number_input_channels, self.number_hidden_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                            nn.BatchNorm2d(self.number_hidden_channels),
                            nn.ReLU(True))
    self.Convolution2 = nn.Sequential(nn.Conv2d(self.number_hidden_channels, self.number_hidden_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                                       nn.BatchNorm2d(self.number_hidden_channels),
                                       nn.ReLU(True))
    self.Convolution3 = nn.Sequential(nn.Conv2d(self.number_hidden_channels, self.number_hidden_channels, kernel_size=3, padding=1, padding_mode='reflect'),
                                       nn.BatchNorm2d(self.number_hidden_channels),
                                       nn.ReLU(True))
    self.Convolution4 = nn.Conv2d(self.number_hidden_channels, 1, kernel_size=3, padding=1, padding_mode='reflect')


    self.Convolution1[0].apply(self.initialize_layer_weights)
    self.Convolution2[0].apply(self.initialize_layer_weights)
    self.Convolution3[0].apply(self.initialize_layer_weights)
    self.Convolution4.apply(self.initialize_layer_weights)

    self.optimizer = optim.Adam(self.parameters(), self.learning_rate)
    
  def initialize_layer_weights(self, layer):
    torch.nn.init.xavier_uniform(layer.weight)
    
  def forward(self, input):
    x1 = self.Convolution1(input)
    x2 = self.Convolution2(x1)
    x3 = self.Convolution3(x2)
    out = self.Convolution4(x3)
    
    return out

