from torch_geometric.nn import GCNConv
import torch
import torch.nn as nn
import utils as functions
from torch.nn import Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Gcn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        # Init parent
        super(Gcn, self).__init__()
        torch.manual_seed(42)
        self.learning_rate = 0.01

        # GCN layers
        self.initial_conv = GCNConv(input_dim, hidden_dim)
        self.conv1 = GCNConv(hidden_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)

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
