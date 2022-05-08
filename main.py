import os
import argparse
import torch

import numpy as np
from pathlib import Path

# for manual visualisation
# from rasterio.plot import show
from dataset import GreenhouseDataset

working_directory = "C:/Users/57834/Documents/thesis"
os.chdir(working_directory)

import utils as functions
import train as train
import evaluate as evaluate
from torch.utils.data import ConcatDataset

import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

import dataset
import dataset_graphs

DO_GRAPH_ANALYSIS = False
DO_CNN_RUN = False
DO_RF_OBIA = False
DO_RF = True

# 2019, 2014, 2011
years = [2019]


def main():
    if DO_CNN_RUN:
        folders_data, folders_labels = dataset.get_data_folders(years)

        datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)

        dataset_full = ConcatDataset(datasets)

        # train_data, test_data = dataset.split_dataset(dataset_full, test_size=0.1)
        train_dataset, test_dataset = dataset.split_dataset_geographically(dataset_full, x_limit=36.523466)

        import model.cnn as CNN
        model = CNN.CnnSemanticSegmentation()
        model.learning_rate = 0.02

        print("starting the training")

        log_message_model = "trying to get some good results and having a baseline"
        folder_results_path = f"runs/CnnModel/{current_time}/"
        os.mkdir(folder_results_path)

        model = train.train(model, train_dataset, number_epochs=30, description_model=log_message_model,
                            results_folder=folder_results_path)

        folder_cnn_save = "runs/CnnModel/"
        evaluate.validation(model, test_dataset, description_model=log_message_model,
                            folder_results=folder_results_path)

    if DO_GRAPH_ANALYSIS:
        folder_graphs, folders_labels, folder_semantic_maps = dataset_graphs.get_data_folders(years)

        dataset_full = dataset_graphs.merge_datasets(folders_labels, folder_graphs, folder_semantic_maps)
        train_dataset, test_dataset = dataset_graphs.split_dataset_geographically(dataset_full, x_limit=36.523466)

        import model.gcn as gcn
        model = gcn.GnnStack(input_dim=13, hidden_dim=128, output_dim=1)
        model.learning_rate = 0.01

        log_message_model = "Testing on a single year to see if we can predict positives."
        folder_results_path = f"runs/Gnn/normal_gcn_{current_time}/"
        os.mkdir(folder_results_path)

        graph_model = train.train_graph(model, train_dataset, number_epochs=40, description_model=log_message_model,
                                        folder_result=folder_results_path)

        evaluate.validation_graph(graph_model, test_dataset, description_model=log_message_model,
                                  folder_results=folder_results_path)

    if DO_RF_OBIA:
        log_message_model = "testing folder saves"
        folder_results_path = f"runs/obia_baseline/{current_time}/"
        os.mkdir(folder_results_path)

        folder_graphs, folders_labels, folder_semantic_maps = dataset_graphs.get_data_folders(years)
        dataset_full = dataset_graphs.merge_datasets(folders_labels, folder_graphs, folder_semantic_maps)
        train_dataset, test_dataset = dataset_graphs.split_dataset_geographically(dataset_full, x_limit=36.523466)

        graph_files_test = dataset_graphs.get_graphs_from_subset_dataset(test_dataset)
        graph_files_train = dataset_graphs.get_graphs_from_subset_dataset(train_dataset)
        x_train, y_train = functions.graphs_to_node_data(graph_files_train)
        x_test, y_test = functions.graphs_to_node_data(graph_files_test)
        evaluate.rf_accuracy_estimation(x_train, y_train, x_test, y_test, description_model=log_message_model,
                                        folder_results=folder_results_path)

    if DO_RF:
        log_message_model = "testing folder saves"
        folder_results_path = f"runs/rf_baseline/{current_time}/"
        os.mkdir(folder_results_path)

        folders_data, folders_labels = dataset.get_data_folders(years)
        datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)
        dataset_full = ConcatDataset(datasets)
        train_dataset, test_dataset = dataset.split_dataset_geographically(dataset_full, x_limit=36.523466)
        x_train, y_train = dataset.get_x_and_y_from_subset_dataset(train_dataset)
        x_test, y_test = dataset.get_x_and_y_from_subset_dataset(test_dataset)
        evaluate.rf_accuracy_estimation(x_train, y_train, x_test, y_test, log_message_model, folder_results_path)


if __name__ == "__main__":
    main()

    """  

    ### training a mlp without grap data (OO),
    #graph_files = dataset_graphs.graph_files
    #x_nodes, y_nodes = functions.graph_files_to_node_data(graph_files)
    # balancing two dataset
    #x_nodes_bal, y_nodes_bal = functions.balancing_binary_dataset(x_nodes, y_nodes)

    # test on a simplified GCN
    #import analysis_and_testing.simplified_gcn as SimpleGcn

    #simple_GCN = SimpleGcn.GCN(num_feature_in=13, hidden_dim=128, num_classes=1)
    #graph_model_simple = train.train_graph(simple_GCN, dataset_graphs, 100)

    #graph_model = GCN.GNNStack(input_dim=13, hidden_dim=32, output_dim=1)
    #trained_model = train.train_graph(graph_model, dataset_graphs, 20)




    # running the vanilla model
    # args, trained_model_vanilla, train_data = main("vanilla", train_data)

    # # training a RF model
    # graph_files = dataset_graphs.graph_files
    # x_nodes, y_nodes = functions.graph_files_to_node_data(graph_files)
    # evaluate.rf_accuracy_estimation(x_nodes, y_nodes)
    
    """

    """
    # evaluation on the vanilla model
    #conf_mat_nn, class_report_nn = evaluate.nn_accuracy_estimation(trained_model_vanilla, test_data)
    
    # evaluation on the vanilla model
    #conf_mat_nn, class_report_nn = evaluate.nn_accuracy_estimation(model_resnet, test_data)
    
    # evaluation on the vanilla model
    #conf_mat_nn, class_report_nn = evaluate.nn_accuracy_estimation(model_deeplab, test_data)
    
    ## working on the baselines
    # extracting data and labels
    train_set = [sample for sample in train_data]
    test = [sample for sample in test_data]
    
    #svm
# =============================================================================
#     conf_mat_svm, class_report_svm, _ = evaluate.svm_accuracy_estimation(train_set, test)
# =============================================================================
    # training a RF model
    conf_mat_rf, class_report_rf, _ = evaluate.rf_accuracy_estimation(train_set, test)
    """

"""
## previous dataset
    subset = False
    train_gt = dataset.get_datasets(transform=True)
    train_data = train_gt
    train_neg = dataset.get_datasets(nairo=True, subset=subset)
    train_neg, test_neg = train_test_split(train_neg, test_size=0.9)
    test_gt, train_gt = train_test_split(train_gt, test_size=0.7)
    # fusing gt and negatives for the datasets
    train_data = torch.utils.data.ConcatDataset([train_neg, train_gt])
    test_data = torch.utils.data.ConcatDataset([test_neg, test_gt])
    
     # running the transfer learning model
    #args, model_deeplab, train_data = main("deeplab", train_data)
    
    # running the transfer learning model
    #args, model_resnet, train_data = main("resnet", train_data)
    
    #args, trained_model_deeplab, _, _ = main("deeplab")
"""

## TODO for random forest do pre processing in a funciton

## creating train and test
# nb_bands, dims = train[0][0].shape[0], train[0][0].shape[1]
# x_train = np.concatenate([samp[0].reshape((nb_bands, dims ** 2)).T for samp in train])
# x_test = np.concatenate([samp[0].reshape((nb_bands, dims ** 2)).T for samp in test])
# y_test = np.concatenate([samp[1] for samp in test])
# y_train = np.concatenate([samp[1] for samp in train])
