
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(CustomConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

        torch.nn.init.xavier_uniform_(self.lin.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.lin_self, gain=0.001)

    def forward(self, x, edge_index, edge_attribute):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Add self-loops to the adjacency matrix.
        edge_index, _ = pyg_utils.add_self_loops(edge_index)

        # Transform node feature matrix.
        self_x = self.lin_self(x)
        #x = self.lin(x)

        # here the linear layer is done on the neighbours
        return self_x + self.propagate(edge_index, edge_attr=edge_attribute, size=(x.size(0), x.size(0)), x=self.lin(x))

    def message(self, x_j, edge_index, size, edge_attr):
      # Constructs messages to node in analogy to for each edge (i, j)
      # Note that we generally refer to as the central nodes that aggregates
      # information, and refer to as the neighboring nodes, since this is the most common notation
        # Compute messages
        # x_j has shape [E, out_channels]
        # _j refers to neighbours
        index_targets, index_neighbours = edge_index
        deg = pyg_utils.degree(index_targets, size[0], dtype=x_j.dtype)
        degree_inverse_sqrt = deg.pow(-0.5)
        norm = degree_inverse_sqrt[index_targets] * degree_inverse_sqrt[index_neighbours]

        return norm.view(-1, 1) * x_j
        #return x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out