import torch
from torch_geometric.data import Data, DataLoader
import MNIST.mnist_utils as functions

class SuperpixelDataset(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.graphs = []

        self.num_classes = 10

        for sample in dataset:
            image = sample[0]
            label = sample[1]

            mask_segmentation = functions.SLIC_segmented_array_from_image(image)
            mask_segmentation -= 1
            adjacency_matrix = functions.adjacency_matrix_from_labelled_image(mask_segmentation)
            edge_index = functions.edge_index_from_adjacency_matrix(adjacency_matrix)
            edge_index = edge_index.long()
            nodes_features, coordinates = functions.nodes_from_image(image, mask_segmentation)

            distance_connected_nodes = functions.extract_distance_edges(coordinates, edge_index)
            graph_sample = Data(x=nodes_features, edge_index=edge_index, y=label,
                                edge_attr=distance_connected_nodes)
            self.graphs.append(graph_sample)

        self.num_node_features = nodes_features.shape[1]

    def __len__(self):
        return len(self.graphs)

    def __getitem__(self, idx):
        return self.graphs[idx]