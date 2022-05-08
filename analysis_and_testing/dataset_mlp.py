# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:03:03 2021

@author: baron015
"""

import torch
from torch.utils.data.dataset import Dataset

import numpy as np
import os
import random

# import functions from other files
import utils as fun


class mlp_graph_dataset(Dataset):

    def __init__(self, nodes, labels):
        self.nodes = nodes
        self.labels = labels

        self.nodes.sort()
        self.labels.sort()

    def __getitem__(self, idx_sample):

        node_x = self.nodes[idx_sample]
        label_y = self.labels[idx_sample]

        node_x = torch.tensor(node_x)
        label_y = torch.tensor(label_y)

        return node_x, label_y

    def __len__(self):
        return len(self.nodes)
