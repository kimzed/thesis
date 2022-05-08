
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn

import MNIST.GraphLayer as GraphLayer
import MNIST.EquivariantLayer as EquivariantLayer

class EquivariantGNNStack(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, task='node'):
        super(EquivariantGNNStack, self).__init__()
        self.task = task
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

        return EquivariantLayer.EquivariantLayer(hidden_dim, hidden_dim)


    def forward(self, data):
        x, edge_index, batch, edge_attributes = data.x, data.edge_index, data.batch, data.edge_attr

        x = self.embedding_in(x)

        for i in range(self.num_layers):
            x = self.convs[i](x, edge_index, edge_attributes, batch)

        if self.task == 'graph':
            x = pyg_nn.global_mean_pool(x, batch)

        x = self.post_mp(x)

        return F.log_softmax(x, dim=1)

    def loss(self, pred, label):
        return F.nll_loss(pred, label)