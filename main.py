# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:30:19 2021

@author: baron015
"""

import os
import argparse
import torch

import numpy as np

# for manual visualisation
#from rasterio.plot import show

working_directory = "//WURNET.NL/Homes/baron015/My Documents/thesis"
os.chdir(working_directory)

import utils as functions
import train as train

import warnings
warnings.filterwarnings('ignore')


def main():



    years = [2019]

    print("""
    
    Training the CNN    

    """)

    import dataset
    folders_data, folders_labels = dataset.get_data_folders(years)

    dataset_full = dataset.merge_datasets(data_folders=folders_data, label_folders=folders_labels)

    train_data, test_data = dataset.split_dataset(dataset_full, test_size=0.1)

    #### subsetting for testing purposes
    indexes_dataset = np.arange(0, train_data.__len__())
    from sklearn.model_selection import train_test_split
    _, indexes_train = train_test_split(indexes_dataset, test_size=0.1, random_state=42)
    train_data = torch.utils.data.Subset(train_data, indexes_train)

    import model.CNN as CNN
    model = CNN.CnnSemanticSegmentation()

    import train
    print("starting the training")

    log_message_model = "normal training for the CNN on partial dataset"
    train.train(model, train_data, number_epochs=4, description_model=log_message_model)


        
if __name__ == "__main__":

    main()




    ### graph data / model
    #folder_greenhouse_dataset = f"data/{year}/greenhouse_dataset/"
    #folder_graphs = os.path.join(folder_greenhouse_dataset, "graphs")
    #folder_semantic_maps = os.path.join(folder_greenhouse_dataset, "semantic_maps_graphs")
    #folder_labels = os.path.join(folder_greenhouse_dataset, "ground_truth_rasters")

    ### graph model
    #dataset_graphs = dataset_graph.GraphDatasetSemanticSegmentation(folder_graphs, folder_labels, folder_semantic_maps)

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
#nb_bands, dims = train[0][0].shape[0], train[0][0].shape[1]
#x_train = np.concatenate([samp[0].reshape((nb_bands, dims ** 2)).T for samp in train])
#x_test = np.concatenate([samp[0].reshape((nb_bands, dims ** 2)).T for samp in test])
#y_test = np.concatenate([samp[1] for samp in test])
#y_train = np.concatenate([samp[1] for samp in train])

