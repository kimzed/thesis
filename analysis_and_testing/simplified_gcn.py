from torch_geometric.nn import GCNConv

# simple GNN layer
from torch_geometric.nn import GraphConv
import torch
import torch.nn.functional as F
import torch.nn as nn

import MNIST.mnist_utils as functions


class GCN(torch.nn.Module):
    def __init__(self, num_feature_in, hidden_dim, num_classes):
        super().__init__()
        self.conv1 = GCNConv(num_feature_in, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, num_classes)
        self.linear = nn.Linear(hidden_dim, num_classes)


    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x

    def predict(self, data):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=0.5)

        return prediction
