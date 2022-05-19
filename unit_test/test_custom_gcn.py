from torch_geometric.datasets import GNNBenchmarkDataset
from model.custom_gcn import Gcn
from model.built_in_gcn import Gcn as BuiltInGcn
import torch
from train import train_graph
from train import UNIT_TEST
import dataset as functions_dataset
import evaluate
from evaluate import UNIT_TEST_EVALUATE
from utils import count_parameters
from torch_geometric.data import Data

dataset = GNNBenchmarkDataset(root="python/unit_test/benchmark_graph_dataset/", name="PATTERN")
folder_results_path = f"python/unit_test/fake_result_runs/"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
UNIT_TEST['is_true'] = True
UNIT_TEST_EVALUATE['is_true'] = True

# test data
x = torch.tensor([[-9999], [9999], [9999], [9999]], dtype=torch.float)
y = torch.tensor([1, 0, 0, 0], dtype=torch.float)
edge_index = torch.tensor([[0, 0, 0],
                           [1, 2, 3]], dtype=torch.long)
data = Data(x=x, y=y, edge_index=edge_index)


def test_gcn_validation_results_on_dummy_dataset_are_good():
    train_dataset, test_dataset = functions_dataset.split_dataset(dataset, test_size=0.2)

    size_hidden_features = 32

    size_input = max(dataset.num_node_features, 1)

    model = Gcn(input_dim=size_input, hidden_dim=size_hidden_features, output_dim=1)

    number_epochs = 3
    description_model = "testing on benchmark dataset, custom model"
    model = train_graph(dataset=train_dataset, model=model, number_epochs=number_epochs,
                        folder_result=folder_results_path,
                        description_model=description_model)

    evaluate.validation_graph(model, test_dataset, description_model=description_model,
                              folder_results=folder_results_path)


def test_gcn_number_parameters_same_as_built_in_model():
    model_custom = Gcn(input_dim=2, hidden_dim=10, output_dim=1)
    model_built_in = BuiltInGcn(input_dim=2, hidden_dim=10, output_dim=1)

    number_parameters_custom = count_parameters(model_custom)
    number_parameters_built_in = count_parameters(model_built_in)

    assert number_parameters_custom == number_parameters_built_in, "number of parameters must be the same between built in gcn and custom replica"


def test_forward_():
    model_custom = Gcn(input_dim=1, hidden_dim=1, output_dim=1)
    model_built_in = BuiltInGcn(input_dim=1, hidden_dim=1, output_dim=1)


    result_custom = model_custom(data)
    result_built_in = model_built_in(data)

def test_gcn_initial_conv_has_similar_parameter_types_as_built_in():
    model_custom = Gcn(input_dim=1, hidden_dim=10, output_dim=1)
    model_built_in = BuiltInGcn(input_dim=1, hidden_dim=10, output_dim=1)

    parameters_custom = model_custom.initial_conv.state_dict()
    parameters_built_in = model_built_in.initial_conv.state_dict()

    names_custom = list(parameters_custom.keys())
    names_custom.sort()
    names_built_in = list(parameters_built_in.keys())
    names_built_in.sort()

    assert names_built_in == names_custom

def test_gcn_inital_conv_has_similar_aggregate_function():


    model_custom = Gcn(input_dim=1, hidden_dim=1, output_dim=1)
    model_built_in = BuiltInGcn(input_dim=1, hidden_dim=1, output_dim=1)

    conv_custom = model_custom.initial_conv
    conv_built_in = model_built_in.initial_conv

    x = torch.tensor([[1], [0], [0], [9999]], dtype=torch.float)
    edge_index = torch.tensor([[3], [3], [0], [0]], dtype=int)
    aggregation_custom = conv_custom.aggregate(inputs=x, index=edge_index)
    aggregation_built_in = conv_built_in.aggregate(inputs=x, index=edge_index)

    assert aggregation_custom == aggregation_built_in

def test_test_gcn_inital_conv_non_connected_nodes_do_not_interact():
    model_custom = Gcn(input_dim=1, hidden_dim=1, output_dim=1)
    model_built_in = BuiltInGcn(input_dim=1, hidden_dim=1, output_dim=1)

    x = torch.tensor([[0], [100], [0], [0]], dtype=torch.float)
    y = torch.tensor([1, 0, 0, 0], dtype=torch.float)
    edge_index = torch.tensor([[0,2,3], [1,1,1]], dtype=torch.long)
    data = Data(x=x, y=y, edge_index=edge_index)

    conv_custom = model_custom.initial_conv
    conv_built_in = model_built_in.initial_conv

    # for the custom, indexes 2 and 3 have an output
    # the custom influences the 0s because it has a randomly intiated bias
    out_custom = conv_custom(data.x, edge_index)
    out_built_in = conv_built_in(data.x, edge_index)

# test_gcn_test_results_on_dummy_dataset_are_good()
test_gcn_validation_results_on_dummy_dataset_are_good()
#test_gcn_initial_conv_has_similar_parameter_types_as_built_in()
#test_gcn_number_parameters_same_as_built_in_model()
# test_forward_()
#test_test_gcn_inital_conv_non_connected_nodes_do_not_interact()
