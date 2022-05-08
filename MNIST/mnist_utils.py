
import torch
from skimage.segmentation import slic
import numpy as np
from skimage.measure import regionprops
from skimage.future import graph
import math

def nodes_from_image(image, mask_segmentation):
    node_features = []
    coordinates = []
    data_np = np.array(image)

    p = regionprops(mask_segmentation+1, intensity_image=data_np)
    g = graph.rag_mean_color(data_np, mask_segmentation)

    number_nodes = len(g.nodes)
    for i_node in range(number_nodes):
        # Value with the mean intensity in the region.
        color = p[i_node]['mean_intensity']
        # Hu moments (translation, scale and rotation invariant).
        invariants = p[i_node]['moments_hu']
        coordinate_center = torch.Tensor(p[i_node]['centroid']).unsqueeze(0)
        feat = torch.cat([torch.Tensor([color]), torch.Tensor(invariants)]).unsqueeze(0)
        node_features.append(feat)
        coordinates.append(coordinate_center[0])

    # stacking up all the features in the batches
    node_features = torch.cat(node_features, dim=0)

    return node_features, coordinates


def nodes_from_image_adds_coordinates(image, mask_segmentation):
    node_features = []
    coordinates = []
    data_np = np.array(image)

    p = regionprops(mask_segmentation + 1, intensity_image=data_np)
    g = graph.rag_mean_color(data_np, mask_segmentation)

    for node in g.nodes:
        # Value with the mean intensity in the region.
        color = p[node]['mean_intensity']
        # Hu moments (translation, scale and rotation invariant).
        invariants = p[node]['moments_hu']
        coordinate_center = torch.Tensor(p[node]['centroid']).unsqueeze(0)
        feat = torch.cat([torch.Tensor([color]), torch.Tensor(invariants), coordinate_center.reshape(2)]).unsqueeze(0)
        node_features.append(feat)
        coordinates.append(coordinate_center[0])

    # stacking up all the features in the batches
    node_features = torch.cat(node_features, dim=0)

    return node_features, coordinates


def SLIC_segmented_array_from_image(image):
    data_np = np.array(image)
    mask_segmentation = slic(data_np, n_segments=25, compactness=0.5, sigma=0.1)

    return mask_segmentation


def adjacency_matrix_from_labelled_image(labelled_image):
    number_nodes = labelled_image.max()+1

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


def adjacency_matrix_fully_connected(labelled_image):
    number_row = labelled_image.shape[0]
    number_columns = labelled_image.shape[1]

    return np.ones((number_row, number_columns))


"""
def adjacency_matrix_from_labelled_image_8_connected(labelled_image):

  number_nodes = labelled_image.max() + 1

  adjacency_matrix = np.zeros([number_nodes]*2)

  # left-right pairs
  adjacency_matrix[labelled_image[:, :-1], labelled_image[:, 1:]] = 1
  # right-left pairs
  adjacency_matrix[labelled_image[:, 1:], labelled_image[:, :-1]] = 1
  # top-bottom pairs
  adjacency_matrix[labelled_image[:-1, :], labelled_image[1:, :]] = 1
  # bottom-top pairs
  adjacency_matrix[labelled_image[1:, :], labelled_image[:-1, :]] = 1

  ## doing the diagonales
  adjacency_matrix[labelled_image[:-1, :-1], labelled_image[1:, 1:]] = 1
  adjacency_matrix[labelled_image[1:, :-1], labelled_image[-1:, 1:]] = 1
  adjacency_matrix[labelled_image[:-1, 1:], labelled_image[1:, :-1]] = 1
  adjacency_matrix[labelled_image[1:, 1:], labelled_image[:-1, :-1]] = 1

  adjacency_matrix -= np.eye(number_nodes)

  return adjacency_matrix
"""


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


def distance_between_two_point(point1, point2):
    x1, y1 = point1[0], point1[1]
    x2, y2 = point2[0], point2[1]
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    return distance