import torch
import torch.nn.functional as F
import torch.nn as nn
import utils as functions
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree
from torch.nn import Parameter
from torch_geometric.nn.inits import zeros

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Gcn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Init parent
        super(Gcn, self).__init__()
        torch.manual_seed(42)
        self.learning_rate = 0.01

        # GCN layers
        self.initial_conv = GcnConv(input_dim, hidden_dim)
        self.conv1 = GcnConv(hidden_dim, hidden_dim)
        self.conv2 = GcnConv(hidden_dim, hidden_dim)
        self.conv3 = GcnConv(hidden_dim, hidden_dim)

        # Output layer
        self.out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch_index = data.x, data.edge_index, data.batch
        # First Conv layer
        hidden = self.initial_conv(x, edge_index)
        hidden = torch.tanh(hidden)

        # Other Conv layers
        hidden = self.conv1(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv2(hidden, edge_index)
        hidden = torch.tanh(hidden)
        hidden = self.conv3(hidden, edge_index)
        hidden = torch.tanh(hidden)

        # Apply a final (linear) classifier.
        out = self.out(hidden)

        return out

    def predict_to_binary(self, data, threshold: float = 0.5):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=threshold)
        if device.type == "cuda":
            prediction = prediction.cuda()
        return prediction


class GcnConv(MessagePassing):
    """
    using this website: https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html
    """

    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.lin = Linear(in_channels, out_channels, bias=False, weight_initializer='glorot')

        # following code from torch geometric, im adding my own biases that i will out outside of propagate
        self.bias = Parameter(torch.Tensor(out_channels))

        # initialization
        self.lin.reset_parameters()
        zeros(self.bias)


    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        x = self.lin(x)

        # Step 3: Compute normalization.
        idxs_source, idxs_neighbours = edge_index
        deg = degree(idxs_neighbours, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[idxs_source] * deg_inv_sqrt[idxs_neighbours]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        out += self.bias

        return out

    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j
