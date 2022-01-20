# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:03:03 2021

@author: baron015
"""

import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split

import numpy as np

# import functions from other files
import utils as fun

class GreenhouseDataset(Dataset):

    def __init__(self, folder_data, folder_labels, transform=None):
        
        self.img_files = fun.get_files(folder_data)
        self.mask_files = fun.get_files(folder_labels)
        self.transform = transform

        self.mask_files.sort()
        self.img_files.sort()
        
    
    def __getitem__(self, idx_sample):
        
        img_path = self.img_files[idx_sample]
        mask_path = self.mask_files[idx_sample]
        
        img = np.load(img_path)
        mask = np.load(mask_path)
        
        if self.transform:
            img, mask = fun.transform_numpy(img, mask)
            
        
        return img.copy(), mask.copy()

    
    def __len__(self):
        
        return len(self.img_files)


def merge_datasets(data_folders:list, label_folders:list)-> torch.utils.data.dataset.Dataset:

    datasets = []

    for data_folder, label_folder in zip(data_folders, label_folders):

        if fun.data_folder_has_positives(data_folder):
            dataset = GreenhouseDataset(folder_data=data_folder, folder_labels=label_folder, transform=True)

        else:
            dataset = GreenhouseDataset(folder_data=data_folder, folder_labels=label_folder, transform=False)

        datasets.append(dataset)

    dataset_out = torch.utils.data.ConcatDataset(datasets)

    return dataset_out


def get_data_folders(years:list)-> tuple:

    folders_data = []
    folders_labels = []

    for year in years:
        folder_nairobi_dataset = f"data/{year}/nairobi_negatives_dataset/"
        folder_positive_dataset = f"data/{year}/greenhouse_dataset/"

        folder_data_nairo = f"{folder_nairobi_dataset}landsat_rasters"
        folders_data.append(folder_data_nairo)
        folder_data_positives = f"{folder_positive_dataset}landsat_rasters"
        folders_data.append(folder_data_positives)

        folder_label_nairo = f"{folder_nairobi_dataset}ground_truth_rasters"
        folders_labels.append(folder_label_nairo)
        folder_labels_positives = f"{folder_positive_dataset}ground_truth_rasters"
        folders_labels.append(folder_labels_positives)

    return folders_data, folders_labels


def split_dataset(dataset:torch.utils.data.dataset.Dataset, test_size:float):

    number_samples = dataset.__len__()
    indexes_dataset = np.arange(0, number_samples)

    indexes_train, indexes_test = train_test_split(indexes_dataset, test_size=test_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, indexes_train)
    test_dataset = torch.utils.data.Subset(dataset, indexes_test)

    return train_dataset, test_dataset