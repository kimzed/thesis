
import torch
import torch.nn as nn
import torch_geometric.nn as pyg_nn
from torch_geometric.typing import Size


class EquivariantLayer(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EquivariantLayer, self).__init__(aggr='add')  # "Add" aggregation.
        act_fn = nn.SiLU()

        self.dropout = nn.Dropout(0.25)
        size_distance = 1
        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels + in_channels + size_distance, out_channels),
            self.dropout,
            act_fn,
            nn.Linear(out_channels, in_channels),
            act_fn)

        self.node_mlp = nn.Sequential(
            nn.Linear(in_channels + in_channels, out_channels),
            self.dropout,
            act_fn,
            nn.Linear(out_channels, in_channels))

        self.node_dim = 0

    def forward(self, x, edge_index, edge_attr, batch):
        hidden_out = self.propagate(edge_index, x=x, edge_attr=edge_attr,
                                    batch=batch)

        return hidden_out

    def propagate(self, edge_index, size: Size = None, **kwargs):
        size = self.__check_input__(edge_index, size)
        hidden_feats = kwargs["x"]
        coll_dict = self.__collect__(self.__user_args__, edge_index, size, kwargs)
        msg_kwargs = self.inspector.distribute('message', coll_dict)
        aggr_kwargs = self.inspector.distribute('aggregate', coll_dict)
        update_kwargs = self.inspector.distribute('update', coll_dict)

        # get messages
        m_ij = self.message(**msg_kwargs)
        edge_attr = kwargs["edge_attr"]

        # update feats if specified
        m_i = self.aggregate(m_ij, **aggr_kwargs)

        hidden_out = self.node_mlp(torch.cat([hidden_feats, m_i], dim=-1))
        hidden_out = kwargs["x"] + hidden_out

        return hidden_out

    def message(self, x_i, x_j, edge_index, size, edge_attr):
        # adding a fake dimension
        edge_attr = edge_attr[:, None]
        message_input = torch.cat([x_j, x_i, edge_attr], dim=-1)
        message_transformed = self.edge_mlp(message_input)

        return message_transformed

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out