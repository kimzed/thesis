import torch.nn as nn
import torch_geometric.nn as pyg_nn
import torch
import torch.nn.functional as F
import torch_geometric.utils as pyg_utils
import utils as functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GnnStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GnnStack, self).__init__()
        self.learning_rate = 0.01

        self.dropout = 0.25
        self.number_layers = 2

        self.convs = nn.ModuleList()
        # first layer has different parameters
        self.convs.append(pyg_nn.GCNConv(input_dim, hidden_dim))
        for _ in range(self.number_layers - 1):
            self.convs.append(CustomConv(hidden_dim, hidden_dim))

        self.post_message_passing = nn.ModuleList()
        for _ in range(self.number_layers - 1):
            self.post_message_passing.append(nn.LayerNorm(hidden_dim))

        self.post_processing = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        # x is the convention for node_features
        x, edge_index, batch, edge_attribute = data.x, data.edge_index, data.batch, data.edge_attr

        i_last_layer = self.number_layers - 1

        for i_layer in range(self.number_layers):
            x = self.convs[i_layer](x, edge_index, edge_attribute)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

            is_last_layer = i_layer == i_last_layer

            if not is_last_layer:
                x = self.post_message_passing[i_layer](x)

        out = self.post_processing(x)

        return out

    def predict(self, data):
        prediction = self.forward(data)
        prediction = nn.Sigmoid()(prediction)
        prediction = functions.converting_probability_array_to_binary(prediction, threshold=0.5)
        if device.type == "cuda":
            prediction = prediction.cuda()
        return prediction


class CustomConv(pyg_nn.MessagePassing):
    def __init__(self, in_channels, out_channels):
        aggregation_method = "add"
        super(CustomConv, self).__init__(aggr=aggregation_method)
        self.lin = nn.Linear(in_channels, out_channels)
        self.lin_self = nn.Linear(in_channels, out_channels)

        torch.nn.init.xavier_uniform_(self.lin.weight, gain=0.001)
        torch.nn.init.xavier_uniform_(self.lin_self.weight, gain=0.001)

    def forward(self, x, edge_index, edge_attribute):
        edge_index, _ = pyg_utils.add_self_loops(edge_index)

        x = self.lin_self(x)

        row, col = edge_index
        deg = pyg_utils.degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        propagation = self.propagate(edge_index=edge_index, x=x,
                                     norm=norm, edge_attr=edge_attribute,
                                     size=(x.size(0), x.size(0)))

        return propagation

    def message(self, x_j, edge_index, size, edge_attr, norm):
        # x_j is the neighbour node

        message = norm.view(-1, 1) * x_j

        return message
