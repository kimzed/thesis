# -*- coding: utf-8 -*-
"""
Created on Sat Jul 24 14:03:03 2021

@author: baron015
"""
import os

import torch
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple
from torch.utils.data import Subset

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
        self.centroids_coordinates = self.get_centroids_coordinates(Path(folder_data))

    def __getitem__(self, idx_sample):
        img_path = self.img_files[idx_sample]
        mask_path = self.mask_files[idx_sample]

        img = np.load(img_path)
        mask = np.load(mask_path)

        if self.transform:
            img, mask = fun.transform_numpy(img, mask)

        x = float(self.centroids_coordinates[idx_sample][0])
        y = float(self.centroids_coordinates[idx_sample][1])
        coordinate_sample = x, y

        return img.copy(), mask.copy(), coordinate_sample

    def get_centroids_coordinates(self, raster_folder_path: Path) -> list[tuple]:
        path_coordinate_folder = raster_folder_path.parent.absolute()
        file_coordinates = "coordinates_rasters.csv"
        path_coordinates = os.path.join(path_coordinate_folder, file_coordinates)

        centroids = fun.read_csv_into_list(path_coordinates)

        return centroids

    def __len__(self):
        return len(self.img_files)


def get_datasets(data_folders: list[str], label_folders: list[str]) -> list[Dataset]:
    datasets = []

    for data_folder, label_folder in zip(data_folders, label_folders):

        if fun.data_folder_has_positives(data_folder):
            dataset = GreenhouseDataset(folder_data=data_folder, folder_labels=label_folder, transform=True)

        else:
            dataset = GreenhouseDataset(folder_data=data_folder, folder_labels=label_folder, transform=False)

        datasets.append(dataset)

    return datasets


def get_data_folders(years: list) -> tuple:
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


def split_dataset(dataset: torch.utils.data.dataset.Dataset, test_size: float):
    number_samples = dataset.__len__()
    indexes_dataset = np.arange(0, number_samples)

    indexes_train, indexes_test = train_test_split(indexes_dataset, test_size=test_size, random_state=42)

    train_dataset = Subset(dataset, indexes_train)
    test_dataset = Subset(dataset, indexes_test)

    return train_dataset, test_dataset


def split_dataset_geographically(dataset: Dataset, x_limit: float) -> Tuple[Dataset, Dataset]:
    x_coordinates = []
    len_dataset = dataset.__len__()

    for i_sample in range(len_dataset):
        sample = dataset.__getitem__(i_sample)
        coordinate = sample[2]
        x_coordinate = coordinate[0]
        x_coordinates.append(x_coordinate)

    x_coordinates_array = np.array(x_coordinates)
    indexes_test = np.where(x_coordinates_array < x_limit)[0]
    indexes_train = np.where(x_coordinates_array > x_limit)[0]

    train_dataset = Subset(dataset, indexes_train)
    test_dataset = Subset(dataset, indexes_test)

    return train_dataset, test_dataset

def get_x_and_y_from_subset_dataset(dataset_mix: torch.utils.data.dataset.Subset)-> Tuple[np.array, np.array]:
    data_arrays = []
    label_arrays = []
    number_samples = dataset_mix.__len__()
    for i_sample in range(number_samples):
        sample = dataset_mix.__getitem__(i_sample)
        data = sample[0]
        labels = sample[1]

        number_bands = data.shape[0]
        image_size = data.shape[1]
        data = data.reshape(number_bands, image_size**2)
        data = np.moveaxis(data, 0, -1)

        label_arrays.append(labels)
        data_arrays.append(data)

    y = np.concatenate(label_arrays)
    x = np.concatenate(data_arrays, axis=0)

    return x, y

