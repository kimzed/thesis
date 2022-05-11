from torch_geometric.nn import GCNConv
import torch
import torch.nn.functional as F
import torch.nn as nn
import utils as functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Gcn(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.learning_rate = 0.01
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.conv3(x, edge_index)

        return x

    def predict(self, data):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=0.5)
        if device.type == "cuda":
            prediction = prediction.cuda()
        return prediction
