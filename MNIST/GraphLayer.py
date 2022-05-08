

import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch_geometric.utils as pyg_utils

class GraphLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GraphLayer, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        edge_index, _ = pyg_utils.remove_self_loops(edge_index)

        # Transform node feature matrix.
        self_x = self.lin_self(x)

        # here the linear layer is done on the neighbours
        return self_x + self.propagate(edge_index, size=(x.size(0), x.size(0)), x=self.lin(x),
                                       edge_attr=edge_attr)

    def message(self, x_j, edge_index, size, edge_attr):
        return x_j  # + edge_attr

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out