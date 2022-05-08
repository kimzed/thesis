"""
Created on Sat Jul 24 14:03:03 2021

@author: baron015
"""

import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import utils as functions
from torch.utils.data import Subset
import os
from pathlib import Path
from typing import Tuple


class GraphDatasetSemanticSegmentation(Dataset):

    def __init__(self, folder_data, folder_labels, folder_segmentation_maps):
        self.graph_files = functions.get_files(folder_data)
        self.mask_files = functions.get_files(folder_labels)
        self.segmentation_map_files = functions.get_files(folder_segmentation_maps)
        self.train = True

        self.mask_files.sort()
        self.graph_files.sort()
        self.segmentation_map_files.sort()
        self.centroids_coordinates = self.get_centroids_coordinates(Path(folder_data))

    def __getitem__(self, idx_sample):
        graph_path = self.graph_files[idx_sample]
        mask_path = self.mask_files[idx_sample]
        segmentation_map_path = self.segmentation_map_files[idx_sample]

        graph = torch.load(graph_path)
        graph = functions.normalize_nodes_features(graph)
        segmentation_map = np.load(segmentation_map_path)
        mask = np.load(mask_path)
        mask = functions.resize_labels_into_image(labels=mask, image=segmentation_map)

        # for the dataloader we need to have tensors
        mask_tensor = torch.tensor(mask.copy())
        segmentation_map_tensor = torch.tensor(segmentation_map.copy())

        x = float(self.centroids_coordinates[idx_sample][0])
        y = float(self.centroids_coordinates[idx_sample][1])
        coordinate_sample = x, y

        return graph, mask_tensor, segmentation_map_tensor, coordinate_sample

    def __len__(self):
        return len(self.graph_files)

    def get_centroids_coordinates(self, raster_folder_path: Path) -> list[tuple]:
        path_coordinate_folder = raster_folder_path.parent.absolute()
        file_coordinates = "coordinates_rasters.csv"
        path_coordinates = os.path.join(path_coordinate_folder, file_coordinates)

        centroids = functions.read_csv_into_list(path_coordinates)

        return centroids


def merge_datasets(label_folders: list, graph_folders: list,
                   semantic_map_folders: list) -> torch.utils.data.dataset.Dataset:
    datasets = []

    for label_folder, graph_folder, semantic_map_folder in zip(label_folders, graph_folders, semantic_map_folders):
        dataset = GraphDatasetSemanticSegmentation(folder_data=graph_folder, folder_labels=label_folder,
                                                   folder_segmentation_maps=semantic_map_folder)

        datasets.append(dataset)

    dataset_out = torch.utils.data.ConcatDataset(datasets)

    return dataset_out


def get_data_folders(years: list) -> tuple:
    folders_labels = []
    folder_graphs = []
    folder_semantic_maps = []

    for year in years:
        folder_nairobi_dataset = f"data/{year}/nairobi_negatives_dataset/"
        folder_positive_dataset = f"data/{year}/greenhouse_dataset/"

        # loading labels ground truth
        folder_label_nairo = f"{folder_nairobi_dataset}ground_truth_rasters"
        folders_labels.append(folder_label_nairo)
        folder_labels_positives = f"{folder_positive_dataset}ground_truth_rasters"
        folders_labels.append(folder_labels_positives)

        # loading graphs
        folder_graphs_nairo = f"{folder_nairobi_dataset}graphs"
        folder_graphs.append(folder_graphs_nairo)
        folder_labels_positives = f"{folder_positive_dataset}graphs"
        folder_graphs.append(folder_labels_positives)

        # loading semantic_maps
        folder_semantic_map_nairo = f"{folder_nairobi_dataset}semantic_maps_graphs"
        folder_semantic_maps.append(folder_semantic_map_nairo)
        folder_semantic_map_positives = f"{folder_positive_dataset}semantic_maps_graphs"
        folder_semantic_maps.append(folder_semantic_map_positives)

    return folder_graphs, folders_labels, folder_semantic_maps


def split_dataset_geographically(dataset: Dataset, x_limit: float) -> Tuple[Dataset, Dataset]:
    x_coordinates = []
    len_dataset = dataset.__len__()

    for i_sample in range(len_dataset):
        sample = dataset.__getitem__(i_sample)
        coordinate = sample[3]
        x_coordinate = coordinate[0]
        x_coordinates.append(x_coordinate)

    x_coordinates_array = np.array(x_coordinates)
    indexes_test = np.where(x_coordinates_array < x_limit)[0]
    indexes_train = np.where(x_coordinates_array > x_limit)[0]

    train_dataset = Subset(dataset, indexes_train)
    test_dataset = Subset(dataset, indexes_test)

    return train_dataset, test_dataset


def get_graphs_from_subset_dataset(dataset_mix: torch.utils.data.dataset.Subset):
    graph_files = []
    number_samples = dataset_mix.__len__()
    for i_sample in range(number_samples):
        sample = dataset_mix.__getitem__(i_sample)
        graph = sample[0]
        graph_files.append(graph)

    return graph_files
