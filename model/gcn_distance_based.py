import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F
from torch_geometric.typing import Size
import torch_geometric.utils as pyg_utils
import utils as functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EquivariantGNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(EquivariantGNNStack, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))

        number_layers = 4
        # self.lns = nn.Linear(input_dim, out_node_nf)
        for l in range(number_layers):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # pre_processing
        self.embedding_in = nn.Linear(input_dim, hidden_dim)

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))

        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        return EquivariantLayer(hidden_dim, hidden_dim)

    def forward(self, data):
        x, edge_index, batch, edge_attributes = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.embedding_in(x)

        for i_layer in range(self.num_layers):
            x = self.convs[i_layer](x, edge_index, edge_attributes, batch)

        out = self.post_mp(x)

        return out

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

    def predict(self, data):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=0.5)
        if device.type == "cuda":
            prediction = prediction.cuda()
        return prediction


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

        index_targets, index_neighbours = edge_index
        nodes_degrees = pyg_utils.degree(index_targets, size[0], dtype=x_j.dtype)
        degree_inverse_sqrt = nodes_degrees.pow(-0.5)
        normalization = degree_inverse_sqrt[index_targets] * degree_inverse_sqrt[index_neighbours]

        return normalization.view(-1, 1) * x_j

        return message_transformed

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out
