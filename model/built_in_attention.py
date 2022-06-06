from torch_geometric.nn import GATv2Conv
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils as functions
from torch.nn import Linear

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class AttentionGnn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(AttentionGnn, self).__init__()
        torch.manual_seed(42)
        self.learning_rate = 0.01
        self.initial_layer = GATv2Conv(input_dim, hidden_dim)
        self.layer1 = GATv2Conv(hidden_dim, hidden_dim)
        self.layer2 = GATv2Conv(hidden_dim, hidden_dim)
        self.out = Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.initial_layer(x=x, edge_index=edge_index)
        x = torch.tanh(x)
        x = F.dropout(x, training=self.training)
        x = self.layer1(x=x, edge_index=edge_index)
        x = torch.tanh(x)
        x = self.layer2(x=x, edge_index=edge_index)
        x = torch.tanh(x)
        x = self.out(x)

        return x

    def predict_to_binary(self, data, threshold: float = 0.5):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=threshold)
        if device.type == "cuda":
            prediction = prediction.cuda()
        return prediction

