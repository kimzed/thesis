
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

class GNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(GNNStack, self).__init__()
        self.task = task
        self.convs = nn.ModuleList()
        self.convs.append(self.build_conv_model(input_dim, hidden_dim))
        self.lns = nn.ModuleList()
        self.lns.append(nn.LayerNorm(hidden_dim))
        self.lns.append(nn.LayerNorm(hidden_dim))
        for l in range(2):
            self.convs.append(self.build_conv_model(hidden_dim, hidden_dim))

        # post-message-passing
        self.post_mp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Dropout(0.25),
            nn.Linear(hidden_dim, output_dim))
        if not (self.task == 'node' or self.task == 'graph'):
            raise RuntimeError('Unknown task.')

        self.dropout = 0.25
        self.num_layers = 3

    def build_conv_model(self, input_dim, hidden_dim):
        # refer to pytorch geometric nn module for different implementation of GNNs.
        if self.task == 'node':
            return pyg_nn.GCNConv(input_dim, hidden_dim)
        else:
            return pyg_nn.GINConv(nn.Sequential(nn.Linear(input_dim, hidden_dim),
                                  nn.ReLU(), nn.Linear(hidden_dim, hidden_dim)))

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        if data.num_node_features == 0:
          x = torch.ones(data.num_nodes, 1)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
            if not i == self.num_layers - 1:
                x = self.lns[i](x)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import SiLU
import utils as functions
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

aggregation_method = 'add'


class Gnn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Gnn, self).__init__()
        torch.manual_seed(42)
        self.aggregation_method = 'add'
        self.learning_rate = 0.01
        self.initial_layer = CustomLayer(input_dim, hidden_dim, aggregation_method=self.aggregation_method)
        self.layer1 = CustomLayer(hidden_dim, hidden_dim, aggregation_method=self.aggregation_method)
        self.layer2 = CustomLayer(hidden_dim, hidden_dim, aggregation_method=self.aggregation_method)
        self.out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        data = self.initial_layer(data)
        data.x = torch.tanh(data.x)
        data.x = F.dropout(data.x, training=self.training)
        data = self.layer1(data)
        data.x = torch.tanh(data.x)
        data = self.layer2(data)
        data.x = torch.tanh(data.x)
        x = pyg_nn.global_mean_pool(data.x, data.batch)
        data.x = self.out(data.x)

        return data.x

    def predict_to_binary(self, data, threshold: float = 0.5):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=threshold)
        if device.type == "cuda":
            prediction = prediction.cuda()
        return prediction


class CustomLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, aggregation_method=aggregation_method):
        # _x refers to source node, while _j refers to neighbour node
        super().__init__(aggr=aggregation_method)

        # optional, adding a bias
        self.bias = Parameter(torch.Tensor(out_channels))

        # setting dropout to 0.8, between 0.5 and 1 is recommended. applying soft dropout to start
        self.dropout = nn.Dropout(0.8)
        self.lin_j = Linear(in_channels, out_channels, bias=True)
        self.lin_x = Linear(in_channels, out_channels, bias=False)

        self.lin_j.reset_parameters()
        self.lin_x.reset_parameters()

        self.edge_mlp = Linear(in_channels + 1, out_channels, bias=False)
        #self.edge_mlp = nn.Sequential(
        #    nn.Linear(in_channels + 1, in_channels * 2),
        #    self.dropout,
        #    SiLU(),
        #    nn.Linear(in_channels * 2, out_channels),
        #    SiLU()
        #)
        self.edge_mlp.reset_parameters()
        #self.edge_mlp.apply(self.init_)

    def init_(self, module):
        """taken from https://github.com/lucidrains/egnn-pytorch/blob/main/egnn_pytorch/egnn_pytorch_geometric.py"""
        if type(module) in {nn.Linear}:
            # seems to be needed to keep the network from exploding to NaN with greater depths
            nn.init.xavier_normal_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, data):
        """
        edge_attribute is distance in our dataset
        """
        if 'edge_attr' in data.keys:
            x, edge_index, batch, edge_attributes = data.x, data.edge_index, data.batch, data.edge_attr
            out_j = self.propagate(edge_index, x=x, edge_attribute=edge_attributes)

        else:
            x, edge_index, batch = data.x, data.edge_index, data.batch
            out_j = self.propagate(edge_index, x=x)
            out_j = self.lin_j(out_j)

        out_x = self.lin_x(x)
        out = out_j + out_x

        # optional, adding a bias
        # out += self.bias

        data.x = out

        return data

    def message(self, x_j, edge_attribute=None):
        # adding a fake dimension
        if edge_attribute != None:
            edge_attr = edge_attribute[:, None]

            message_input = self.edge_mlp(torch.cat([x_j, edge_attr], dim=-1))
            return message_input
        else:
            return x_j
