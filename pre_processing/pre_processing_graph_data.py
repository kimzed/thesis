# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 15:16:58 2021

@author: baron015
"""

import os
from numpy import load
import numpy as np
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from random import randint
import torch
from skimage.measure import regionprops
from skimage.future import graph
import math
from torch_geometric.data import Data
from skimage.measure import regionprops
from skimage.future import graph
from skimage.segmentation import slic

from torch_geometric.data import Data, DataLoader


working_directory = "//WURNET.NL/Homes/baron015/My Documents/thesis"
os.chdir(working_directory)

import utils as functions

def main():

    print("""
          
          Loading the dataset
          
          """)

    year = 2019

    folder_nairobi_dataset = f"data/{year}/nairobi_negatives_dataset/"
    folder_greenhouse_dataset = f"data/{year}/greenhouse_dataset/"

    folder_gt = "ground_truth_rasters/"
    folder_landsat = "landsat_rasters/"

    folder_semantic_maps = "semantic_maps_graphs/"
    folder_graphs = "graphs/"

    files_gt = functions.get_files(folder_greenhouse_dataset + folder_gt)
    files_landsat = functions.get_files(folder_greenhouse_dataset + folder_landsat)




    print("""
          
          Computing the graph data
          
          """)

    graphs, segmentation_maps = semantic_segmentation_dataset_to_graph_dataset(files_landsat, files_gt)

    print("""
    
        saving graphs and segmentation maps
        
        """)

    len_dataset = len(files_landsat)

    path_semantic_map = folder_greenhouse_dataset + folder_semantic_maps
    path_graphs = folder_greenhouse_dataset + folder_graphs

    functions.create_directory(path_semantic_map)
    functions.create_directory(path_graphs)

    for index_sample in range(len_dataset):

        file_name = get_file_from_path(files_landsat[index_sample])
        number_file = get_numbers_from_string(file_name)
        segmentation_map = segmentation_maps[index_sample]
        graph = graphs[index_sample]



        filename_graph = f"{path_graphs}graph{number_file}"
        filename_semantic_map = f"{path_semantic_map}semantic_map{number_file}"


        torch.save(graph, filename_graph)
        np.save(filename_semantic_map, segmentation_map)



def get_file_from_path(path: str)->str:

    elements_file = path.split("/")
    filename = elements_file[-1]

    return filename


def semantic_segmentation_dataset_to_graph_dataset(images: list, segmentation_masks: list):

    graphs = []
    segmentation_maps = []

    for image_file, mask_file in zip(images, segmentation_masks):

        array_image = load(image_file)
        array_mask = load(mask_file)
        # for landsat images we need to put the band at the end
        array_image = functions.put_bands_in_last_dimension(array_image)

        graph, segmentation_map = make_graph_from_image(array_image, array_mask)
        graphs.append(graph)
        segmentation_maps.append(segmentation_map)


    return graphs, segmentation_maps


def edge_index_from_segmentation_mask(segmentation_mask: np.array):

    adjacency_matrix = adjacency_matrix_from_labelled_image(segmentation_mask)
    edge_index = edge_index_from_adjacency_matrix(adjacency_matrix)

    return edge_index


def make_graph_from_image(image, labels):


    labels_image = functions.resize_labels_into_image(labels, image)


    mask_segmentation = slic(image, n_segments=40, compactness=0.001, sigma=3,
                             multichannel=True)

    # first class needs to be zero
    if 0 not in mask_segmentation:
        mask_segmentation -= 1

    label_nodes = get_node_label(labels_image, mask_segmentation)
    adjacency_matrix = adjacency_matrix_from_labelled_image(mask_segmentation)
    edge_index = edge_index_from_adjacency_matrix(adjacency_matrix)
    nodes_features, coordinates = nodes_features_from_image(image, mask_segmentation)
    distance_connected_nodes = extract_distance_edges(coordinates, edge_index)
    graph = Data(x=nodes_features, edge_index=edge_index, y=label_nodes,
                 edge_attr=distance_connected_nodes)

    return graph, mask_segmentation


def get_node_label(image_label: np.array, superpixel_idx_matrix: np.array):

    y_nodes = []
    # first class is zero so we need to add one
    number_nodes = superpixel_idx_matrix.max() +1

    for idx_node in range(number_nodes):

        node_selection = superpixel_idx_matrix == idx_node
        labels_node = image_label[node_selection]
        label_node = get_most_frequent_int_value(labels_node)

        y_nodes.append(label_node)

    y_nodes_tensor = torch.tensor(y_nodes)

    return y_nodes_tensor

def get_most_frequent_int_value(array: np.array):

    counts = np.bincount(array.astype(int))
    most_frequent_value = np.argmax(counts)

    return most_frequent_value

def nodes_features_from_image(image, mask_segmentation):

    node_features = []
    coordinates = []
    data_np = np.array(image)

    p = regionprops(mask_segmentation+1, intensity_image=data_np)
    g = graph.rag_mean_color(data_np, mask_segmentation)

    for idx_node, node in enumerate(g.nodes):

        # Value with the mean intensity in the region.
        color = p[idx_node]['mean_intensity']

        # Hu moments (translation, scale and rotation invariant).
        invariants = p[idx_node]['moments_hu']
        coordinate_center = torch.Tensor(p[idx_node]['centroid']).unsqueeze(0)
        feat = torch.cat([torch.Tensor(color), torch.Tensor(invariants)]).unsqueeze(0)
        node_features.append(feat)
        coordinates.append(coordinate_center[0])

    # stacking up all the features in the batches
    node_features = torch.cat(node_features, dim=0)

    return node_features, coordinates


def adjacency_matrix_from_labelled_image(labelled_image):
    number_nodes = labelled_image.max() + 1

    adjacency_matrix = np.zeros([number_nodes] * 2)

    # left-right pairs
    adjacency_matrix[labelled_image[:, :-1], labelled_image[:, 1:]] = 1
    # right-left pairs
    adjacency_matrix[labelled_image[:, 1:], labelled_image[:, :-1]] = 1
    # top-bottom pairs
    adjacency_matrix[labelled_image[:-1, :], labelled_image[1:, :]] = 1
    # bottom-top pairs
    adjacency_matrix[labelled_image[1:, :], labelled_image[:-1, :]] = 1

    adjacency_matrix -= np.eye(number_nodes)

    return adjacency_matrix


def edge_index_from_adjacency_matrix(adjacency_matrix):
    index_nonzeros = np.array(np.nonzero(adjacency_matrix))
    edge_index = torch.tensor(index_nonzeros)

    return edge_index


def extract_distance_edges(coordinates, edge_index):
    number_edges = edge_index.shape[1]
    distances_edges = torch.zeros(number_edges)

    for i_edge in range(number_edges):
        edge = edge_index[:, i_edge]
        i_target = edge[0]
        i_neighbour = edge[1]
        coordinate_target = coordinates[i_target]
        coordinate_neighbour = coordinates[i_neighbour]
        distance_edge = distance_between_two_point(coordinate_target,
                                                   coordinate_neighbour)
        distances_edges[i_edge] = distance_edge

    return distances_edges


def distance_between_two_point(point1: tuple, point2: tuple):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance

def get_numbers_from_string(string):

    number = ""

    for char in string:

        if char.isdigit():

            number += char

    number = int(number)

    return number



def visualize_graph_sample_from_data_folders(folder_graphs, folder_segmentation_maps, folder_labels, folder_imgs):
    """
    param: a raster 2*128*128, with dem and radiometry
    fun: visualize a given raster in two dimensions and in 3d for altitude
    """

    segmentation_mask, superpixels, mask, img, graph, image_nodes_labels = get_random_graph_data(folder_graphs,
                                                                                                 folder_segmentation_maps, folder_labels, folder_imgs)

    img = img[[3, 2, 1], :, :]
    if np.any(img < 0):
        img = img + np.absolute(img.min())
    img *= (1 / img.max())
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)

    print(f"information about the graph:\n{graph}")
    number_positives = functions.count_value(graph.y, 1)
    number_negatives = functions.count_value(graph.y, 0)
    print(f"graph has {number_positives} positives and {number_negatives} negatives")

    fig, ((rgb, segmentation), (superpixels_ax, nodes_image)) = plt.subplots(2, 2, figsize=(14, 14))

    rgb.set_title("RGB landsat image")
    superpixels_ax.set_title("Superpixels")
    nodes_image.set_title("Nodes label")
    segmentation.set_title("Ground truth labels")

    # removing axis
    segmentation.axis("off")
    superpixels_ax.axis("off")
    nodes_image.axis("off")
    rgb.axis("off")

    # showing the data
    rgb = rgb.imshow(img)
    segmentation = segmentation.imshow(mask)
    superpixels_ax = superpixels_ax.imshow(superpixels)
    nodes_image = nodes_image.imshow(image_nodes_labels)

    plt.show()

def get_random_graph_data(folder_graphs, folder_segmentation_maps, folder_labels, folder_imgs):


    files_graphs = functions.get_files(folder_graphs)
    files_segmentation_maps = functions.get_files(folder_segmentation_maps)
    files_labels = functions.get_files(folder_labels)
    files_rasters = functions.get_files(folder_imgs)

    len_data = len(files_graphs)
    idx_sample = randint(0, len_data)

    segmentation_mask = np.load(files_labels[idx_sample])
    superpixels = np.load(files_segmentation_maps[idx_sample])
    mask = functions.resize_labels_into_image(segmentation_mask, superpixels)
    img = np.load(files_rasters[idx_sample])
    graph = torch.load(files_graphs[idx_sample])
    image_nodes_labels = functions.graph_labels_to_image(graph.y, superpixels)

    return segmentation_mask, superpixels, mask, img, graph, image_nodes_labels


def visualize_random_edge_index(folder_graphs, folder_semantic_maps, folder_labels, folder_imgs):
    segmentation_mask, superpixels, mask, img, graph, image_nodes_labels = get_random_graph_data(folder_graphs,
                                                                                                 folder_semantic_maps,
                                                                                                 folder_labels,
                                                                                                 folder_imgs)

    edge_index = graph.edge_index

    number_nodes = superpixels.max()
    idx_node = randint(0, number_nodes)
    print(f"index number {idx_node}")

    neighbours_idx = get_connected_nodes_idx(idx_node, edge_index)

    map_connected_nodes = superpixels.copy()
    map_connected_nodes[map_connected_nodes == idx_node] = 999

    for idx_neighbour in neighbours_idx:
        map_connected_nodes[map_connected_nodes == idx_neighbour] = 500

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 14))
    ax1.axis("off")
    ax2.axis("off")
    ax1.set_title("Super pixels map")
    ax2.set_title("Origin node and its neighbours")
    ax1 = ax1.imshow(superpixels)
    ax2 = ax2.imshow(map_connected_nodes)
    plt.show()


def get_connected_nodes_idx(idx_node: int, edge_index) -> np.array:
    idx_origins = edge_index[0]
    idx_neighbours = edge_index[1]
    origin_is_node = idx_origins == idx_node
    connected_neighbours = idx_neighbours[origin_is_node]

    return np.array(connected_neighbours)


if __name__ == '__main__':
    main()
