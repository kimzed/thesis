# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:03:03 2021

@author: baron015
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import utils as functions

class GraphDatasetSemanticSegmentation(Dataset):

    def __init__(self, folder_data, folder_labels, folder_segmentation_maps, folder_images=None):
        self.graph_files = functions.get_files(folder_data)
        self.mask_files = functions.get_files(folder_labels)
        self.segmentation_map_files = functions.get_files(folder_segmentation_maps)
        self.train = True

        self.mask_files.sort()
        self.graph_files.sort()
        self.segmentation_map_files.sort()

    def __getitem__(self, idx_sample):
        graph_path = self.graph_files[idx_sample]
        mask_path = self.mask_files[idx_sample]
        segmentation_map_path = self.segmentation_map_files[idx_sample]


        graph = torch.load(graph_path)
        graph = functions.normalize_nodes_features(graph)
        segmentation_map = np.load(segmentation_map_path)
        mask = np.load(mask_path)
        mask = functions.resize_labels_into_image(mask, segmentation_map)


        if self.train:
            return graph
        else:
            return graph, mask.copy(), segmentation_map.copy()

    def __len__(self):
        return len(self.graph_files)
