
from torchvision import datasets
import matplotlib.pyplot as plt
from torch_geometric.data import Data
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import MNIST.mnist_utils as functions

## test on one sample checking if the pre processing works
mnist_train = datasets.MNIST('../data', train=True, download=True)
image = mnist_train[0][0]
label = mnist_train[0][1]

print(f"label is {label}")

plt.imshow(image)
plt.show()

mask_segmentation = functions.SLIC_segmented_array_from_image(image)

plt.imshow(mask_segmentation)
plt.show()

adjacency_matrix = functions.adjacency_matrix_from_labelled_image(mask_segmentation-1)
edge_index = functions.edge_index_from_adjacency_matrix(adjacency_matrix)
edge_index = edge_index.long()

print("\n\n Showing the edge index \n\n")
print(edge_index)
print("\n\n")

nodes_features, coordinates = functions.nodes_from_image(image, mask_segmentation)

print("\n\n Showing the coordinates \n\n")
print(coordinates)
print("\n\n")

## man intensity, moments hue has 7 attributes
plt.imshow(mask_segmentation)
plt.show()
print("\n\n Showing the node features \n\n")
print(nodes_features[:,0])
print("\n\n")

distance_connected_nodes = functions.extract_distance_edges(coordinates, edge_index)
graph_sample = Data(x=nodes_features, edge_index=edge_index, y=label,
              edge_attr=distance_connected_nodes)

print("\n\n Showing the graph data for nodes 0 and 5 \n\n")
print("edge index")
print(graph_sample.edge_index)
print("node 0")
print(f"mean value is {graph_sample.x[0][0]}")
print("node 5")
print(f"mean value is {graph_sample.x[5][0]}")
print("\n\n")
