from torch_geometric.datasets import TUDataset
from train import train_graph
from model.basic_gnn import Gcn
from torch_geometric.data import Data, DataLoader
import torch.optim as optim
import torch
import torch.nn.functional as F
import torch.nn as nn

#dataset = Planetoid(root='/tmp/cora', name='cora')
dataset = TUDataset(root='/tmp/reddit', name='PROTEINS', cleaned=True)
folder_results_path = f"unit_test/fake_result_runs/"
criterion = torch.nn.BCELoss()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_gcn_results_on_dummy_dataset_are_good():
    size_hidden_features = 32

    size_input = max(dataset.num_node_features, 1)

    model = Gcn(input_dim=size_input, hidden_dim=size_hidden_features, output_dim=dataset.num_classes)
    model = train(dataset, model)




def train(dataset, model):
    loader = DataLoader(dataset, batch_size=64, shuffle=True)
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=model.learning_rate)

    for epoch in range(300):
        total_loss = 0
        model.train()
        for batch in loader:
            batch = batch.to(device)

            pred = model(batch)
            label = batch.y
            pred = F.log_softmax(pred, dim=-1)
            pred = pred[batch.train_mask]
            label = label[batch.train_mask]

            # negative log likelihood
            loss = F.nll_loss(pred, label)
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()
            total_loss += loss.item() * batch.num_graphs
        total_loss /= len(loader.dataset)

        if epoch % 10 == 0:
            test_acc = test(loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(
                epoch, total_loss, test_acc))

    return model


def test(loader, model, is_validation=False):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)
        with torch.no_grad():
            pred = model(data)
            pred = pred.argmax(dim=1)

        mask = data.val_mask if is_validation else data.test_mask
        # node classification: only evaluate on nodes in test set
        pred = pred[mask]
        label = data.y[mask]

        correct += pred.eq(label).sum().item()

    total = 0
    for data in loader.dataset:
        total += torch.sum(data.test_mask).item()

    return correct / total

test_gcn_results_on_dummy_dataset_are_good()