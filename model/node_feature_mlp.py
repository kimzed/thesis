
import torch
from torch.nn import Linear
from torch_geometric.data import Data
from torch.nn import ReLU

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Mlp(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(Mlp, self).__init__()
        torch.manual_seed(42)
        self.learning_rate = 0.01
        self.lin1 = Linear(input_dim, hidden_dim)
        self.lin2 = Linear(hidden_dim, hidden_dim*2)
        self.lin3 = Linear(hidden_dim*2, hidden_dim)
        self.out = Linear(hidden_dim, output_dim)

    def forward(self, data:Data):
        x = data.x

        x = self.lin1(x)
        x = ReLU()(x)
        x = self.lin2(x)
        x = ReLU()(x)
        x = self.lin3(x)
        x = ReLU()(x)
        x = self.out(x)

        return x