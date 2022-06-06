import os

working_directory = "C:/Users/57834/Documents/thesis"
os.chdir(working_directory)

import utils as functions
import train as train
import evaluate as evaluate
from evaluate import accuracy_metrics_report
from torch.utils.data import ConcatDataset

import warnings

warnings.filterwarnings('ignore')

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

import dataset
import dataset_graphs

DO_GRAPH_ANALYSIS = True
DO_CNN_RUN = False
DO_RF_OBIA = False
DO_RF = False
SLIC_ACCURACY = False
POSITIVES_SAMPLES_ONLY = True
FULLY_CONNECTED = True
PIXEL_WISE_GRAPH = False

# if we evaluate timewise
years_train = [2019, 2014, 2011]
years_test = [2019, 2014, 2011]


def main():
    if SLIC_ACCURACY:
        accuracy_slic = evaluate.accuracy_slic_segmentation()
        print(f"Accuracy of slic segmentation for graph dataset is {accuracy_slic}")

    if DO_CNN_RUN:
        train_dataset, test_dataset = dataset.get_train_and_test_datasets(years_train, years_test,
                                                                          POSITIVES_SAMPLES_ONLY)

        import model.cnn as CNN
        model = CNN.CnnSemanticSegmentation()
        model.learning_rate = 0.02

        print("starting the training")

        log_message_model = f"new auc for results. lr is {model.learning_rate}"
        folder_results_path = f"runs/CnnModel/new_results_time_analysis_positives_only_{POSITIVES_SAMPLES_ONLY}_{current_time}/"
        os.mkdir(folder_results_path)

        model = train.train(model, train_dataset, number_epochs=100, description_model=log_message_model,
                            results_folder=folder_results_path)

        evaluate.validation(model, test_dataset, description_model=log_message_model,
                            folder_results=folder_results_path)

    if PIXEL_WISE_GRAPH:
        import dataset_pixel_graph
        train_dataset, test_dataset = dataset_pixel_graph.get_train_and_test_datasets(years_train, years_test,
                                                                                 POSITIVES_SAMPLES_ONLY)
        import model.built_in_gnn as gnn
        hidden_dim = 32

        model = gnn.Gnn(input_dim=13, hidden_dim=hidden_dim, output_dim=1)
        model.learning_rate = 0.01

        name_model = type(model).__name__

        log_message_model = f"pixel wise. hidden_dim={hidden_dim}"
        folder_results_path = f"runs/Gnn/pixel_wise_{name_model}_positives_{POSITIVES_SAMPLES_ONLY}_fully_connected_{FULLY_CONNECTED}_{current_time}/"
        os.mkdir(folder_results_path)

        graph_model = train.train_graph(model, train_dataset, number_epochs=100, description_model=log_message_model,
                                        folder_result=folder_results_path)

        evaluate.validation_graph(graph_model, test_dataset, description_model=log_message_model,
                                  folder_results=folder_results_path)



    if DO_GRAPH_ANALYSIS:
        train_dataset, test_dataset = dataset_graphs.get_train_and_test_datasets(years_train, years_test,
                                                                                 POSITIVES_SAMPLES_ONLY,
                                                                                 FULLY_CONNECTED)

        import model.gnn_with_distance as gnn
        hidden_dim = 32
        # we add coordinates as an extra dimension
        model = gnn.DistanceGnn(input_dim=13, hidden_dim=hidden_dim, output_dim=1)
        model.learning_rate = 0.01

        name_model = type(model).__name__

        log_message_model = f"back to normal. hidden_dim={hidden_dim}"
        folder_results_path = f"runs/Gnn/normal_{name_model}_positives_{POSITIVES_SAMPLES_ONLY}_fully_connected_{FULLY_CONNECTED}_{current_time}/"
        os.mkdir(folder_results_path)

        graph_model = train.train_graph(model, train_dataset, number_epochs=100, description_model=log_message_model,
                                        folder_result=folder_results_path)

        evaluate.validation_graph(graph_model, test_dataset, description_model=log_message_model,
                                  folder_results=folder_results_path)

    if DO_RF_OBIA:
        train_dataset, test_dataset = dataset_graphs.get_train_and_test_datasets(years_train, years_test,
                                                                                 POSITIVES_SAMPLES_ONLY,
                                                                                 FULLY_CONNECTED)

        graphs_train, labels_train, semantic_maps_train = dataset_graphs.get_graphs_from_subset_dataset(train_dataset)
        x_train, y_train = functions.graphs_to_node_data(graphs_train)
        rf_model = evaluate.train_rf_model(x_train, y_train)

        prediction_total, y_data_total = evaluate.predict_on_graph_dataset_rf(model=rf_model, dataset=test_dataset)
        folder_results_path = f"runs/obia_baseline/obia_positives_{POSITIVES_SAMPLES_ONLY}_ls8_{current_time}//"
        os.mkdir(folder_results_path)
        report = accuracy_metrics_report(y_data_total, prediction_total)
        file_results = f"{folder_results_path}results_model_{current_time}.txt"
        functions.write_text_file(path=file_results, text=report)

    if DO_RF:
        train_dataset, test_dataset = dataset.get_train_and_test_datasets(years_train, years_test,
                                                                          POSITIVES_SAMPLES_ONLY)
        x_train, y_train = dataset.get_x_and_y_from_subset_dataset(train_dataset)
        x_test, y_test = dataset.get_x_and_y_from_subset_dataset(test_dataset)

        log_message_model = "training on one year and testing on another"
        folder_results_path = f"runs/rf_baseline/rf_positives_{POSITIVES_SAMPLES_ONLY}_ls7_{current_time}/"
        os.mkdir(folder_results_path)
        evaluate.rf_accuracy_estimation(x_train, y_train, x_test, y_test, log_message_model, folder_results_path)


if __name__ == "__main__":
    main()
