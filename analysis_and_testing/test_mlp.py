

# loading required packages
import os
import argparse
import torch
import numpy as np
import torch_geometric


# for manual visualisation
#from rasterio.plot import show



## changing working directory
os.chdir("//WURNET.NL/Homes/baron015/My Documents/thesis")

# importing our functions
import utils as functions
import train as train
import evaluate
import analysis_and_testing.dataset_mlp as dataset_graph_mlp
import analysis_and_testing.MLP_graph_data as ModelMlp



def main():

    year = 2019
    folder_greenhouse_dataset = f"data/{year}/greenhouse_dataset/"
    folder_graphs = os.path.join(folder_greenhouse_dataset, "graphs")
    folder_semantic_maps = os.path.join(folder_greenhouse_dataset, "semantic_maps_graphs")
    folder_labels = os.path.join(folder_greenhouse_dataset, "ground_truth_rasters")

    import dataset_graphs as dataset_graph
    dataset_graphs = dataset_graph.GraphDatasetSemanticSegmentation(folder_graphs, folder_labels, folder_semantic_maps)

    graph_files = dataset_graphs.graph_files
    x_nodes, y_nodes = functions.graph_files_to_node_data(graph_files)
    dataset = dataset_graph_mlp.mlp_graph_dataset(x_nodes, y_nodes)

    #model = ModelMlp.MLP(num_feature_in=13, hidden_dim=64, num_classes=1)
    #train.train(model, dataset, 50)

    model = ModelMlp.MLP(num_feature_in=13, hidden_dim=64, num_classes=1, is_graph_model=True)
    train.train_graph(model, dataset_graphs, 50)

main()