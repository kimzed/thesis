import numpy as np

import dataset_graphs
from utils import graph_labels_to_image

import matplotlib.pyplot as plt


years = [2019, 2014, 2011]

def test_graph_labels_to_image_image_is_correct():
    folder_graphs, folders_labels, folder_semantic_maps = dataset_graphs.get_data_folders(years,
                                                                                          rasters_with_positives_only=True)
    dataset_full = dataset_graphs.merge_datasets(folders_labels, folder_graphs, folder_semantic_maps)

    graph, mask_tensor, segmentation_map, _ = dataset_full.__getitem__(-20)

    mask_expected = np.zeros(segmentation_map.shape)

    for i_node, label in enumerate(graph.y):

        indexes_node = segmentation_map == i_node
        mask_expected[indexes_node] = label


    mask_actual = graph_labels_to_image(graph.y, segmentation_map)

    assert (np.array_equal(mask_expected, mask_actual))

