from torch_geometric.nn import GCNConv

# simple GNN layer
from torch_geometric.nn import GraphConv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils

import utils as functions


class MLP(torch.nn.Module):
    def __init__(self, num_feature_in, hidden_dim, num_classes, is_graph_model=False):
        super().__init__()
        self.lin1 = nn.Sequential(nn.Linear(num_feature_in, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.lin2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.lin3 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU())
        self.lin4 = nn.Linear(hidden_dim, num_classes)

        torch.nn.init.xavier_uniform(self.lin1[0].weight)
        torch.nn.init.xavier_uniform(self.lin2[0].weight)
        torch.nn.init.xavier_uniform(self.lin3[0].weight)
        torch.nn.init.xavier_uniform(self.lin4.weight)

        self.optimizer = optim.Adam(self.parameters(), lr=0.01)

        self.is_graph_model = is_graph_model

    def forward(self, data):

        if self.is_graph_model:
            data = data.x

        x = self.lin1(data)
        x = self.lin2(x)
        x = self.lin3(x)
        x = self.lin4(x)

        return x

    def predict(self, data):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=0.5)

        return prediction


