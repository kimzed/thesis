import torch
import torch.nn.functional as F
import torch.nn as nn
import utils as functions
from torch.nn import Linear
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree, remove_self_loops

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
    based on methodology from this tutorial: https://www.youtube.com/watch?v=-UjytpbqX4A
    for the final code the video has to be checked and the code updated
    """
    def __init__(self, in_channels, out_channels):
        super(GcnConv, self).__init__(aggr='add')  # "Add" aggregation.
        self.lin = nn.Linear(in_channels, out_channels)

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # TODO check in the video what the options for self loop and linear layer are
        # Add self-loops to the adjacency matrix.
        #edge_index, _ = remove_self_loops(edge_index)
        edge_index, _ = add_self_loops(edge_index)

        # Transform node feature matrix.
        #self_x = self.lin_self(x)
        x = self.lin(x)

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j, edge_index, size):
        # Compute messages
        # aggregates all the messages
        # x_j has shape [E, out_channels]

        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return norm.view(-1, 1) * x_j

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]
        return aggr_out