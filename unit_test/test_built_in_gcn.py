from torch_geometric.datasets import GNNBenchmarkDataset
from model.built_in_gcn import Gcn
import torch
from train import train_graph
from train import UNIT_TEST
import dataset as functions_dataset
import evaluate
from evaluate import UNIT_TEST_EVALUATE
from utils import count_parameters

dataset = GNNBenchmarkDataset(root="python/unit_test/benchmark_graph_dataset/", name="PATTERN")
folder_results_path = f"python/unit_test/fake_result_runs/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNIT_TEST['is_true'] = True
UNIT_TEST_EVALUATE['is_true'] = True



def test_gcn_validation_results_on_benchmark_dataset_are_good():

    train_dataset, test_dataset = functions_dataset.split_dataset(dataset, test_size=0.2)

    size_hidden_features = 32

    size_input = max(dataset.num_node_features, 1)

    model = Gcn(input_dim=size_input, hidden_dim=size_hidden_features, output_dim=1)

    number_epochs = 3
    description_model = "testing on benchmark dataset, built-in model. trying to have lower number of layers"
    model = train_graph(dataset=train_dataset, model=model, number_epochs=number_epochs, folder_result=folder_results_path,
                        description_model=description_model)

    evaluate.validation_graph(model, test_dataset, description_model=description_model,
                              folder_results=folder_results_path)


#test_gcn_test_results_on_benchmark_dataset_are_good()
test_gcn_validation_results_on_benchmark_dataset_are_good()
