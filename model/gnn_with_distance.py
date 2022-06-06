import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import SiLU
import utils as functions
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch.nn import Parameter
from torch_geometric.nn.norm import batch_norm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

aggregation_method = 'add'


class DistanceGnn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(DistanceGnn, self).__init__()
        torch.manual_seed(42)
        self.aggregation_method = 'add'
        self.learning_rate = 0.01
        self.initial_layer = CustomLayer(input_dim, hidden_dim, aggregation_method=self.aggregation_method)
        self.layer1 = CustomLayer(hidden_dim, hidden_dim, aggregation_method=self.aggregation_method)
        # self.layer2 = CustomLayer(hidden_dim, hidden_dim, aggregation_method=self.aggregation_method)
        self.out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        # from forum recommendation, testing adding coordinate as an extra feature
        # data.x = torch.cat([data.x, data.pos], dim=-1)

        data = self.initial_layer(data)
        data.x = torch.tanh(data.x)

        data.x = F.dropout(data.x, training=self.training)

        data = self.layer1(data)
        data.x = torch.tanh(data.x)
        # data = self.layer2(data)
        # data.x = torch.tanh(data.x)

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

        # setting dropout to 0.3, between 0 and 0.5 is recommended. applying soft dropout to start
        self.dropout = nn.Dropout(0.3)
        self.lin_j = Linear(in_channels, out_channels, bias=True)
        self.lin_x = Linear(in_channels, out_channels, bias=False)

        self.lin_j.reset_parameters()
        self.lin_x.reset_parameters()

        self.edge_mlp = nn.Sequential(
            nn.Linear(in_channels * 2 + 1, in_channels * 2),
            #self.dropout,
            SiLU(),
            nn.Linear(in_channels * 2, out_channels)
        )
        self.edge_mlp.apply(self.init_)

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

        # option to use only the propagation for the output
        # out = out_j

        # optional, adding a bias
        # out += self.bias

        data.x = out

        return data

    def message(self, x_i, x_j, edge_attribute=None):
        # adding a fake dimension
        if edge_attribute != None:
            edge_attr = edge_attribute[:, None]

            message_input = self.edge_mlp(torch.cat([x_i, x_j, edge_attr], dim=-1))
            return message_input
        else:
            return x_j
