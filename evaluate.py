# -*- coding: utf-8 -*-
"""
Created on Thu Aug 26 19:01:20 2021

@author: baron015
"""
import torch_geometric
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn import metrics
import datetime

# importing useful functions
import dataset
import utils as functions

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
validation = True

now = datetime.datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")


def validation(model: nn.Module, dataset: torch.utils.data.dataset, description_model: str,
               folder_results: str) -> None:
    model = model.eval()

    prediction, y_data, binary_prediction = predict_on_dataset(model, dataset)

    auc = metrics.roc_auc_score(y_data, prediction)
    report = accuracy_metrics_report(y_data, binary_prediction)

    learning_rate = model.learning_rate
    results_to_save = f" learning rate= {learning_rate}\n\n" + description_model + f"\n\n auc= {auc}"
    results_to_save += f"\n\n {report}"

    file_results = f"{folder_results}results_model_{current_time}.txt"

    functions.write_text_file(path=file_results, text=results_to_save)


def predict_on_dataset(model: nn.Module, dataset: torch.utils.data.dataset):
    loader = DataLoader(dataset, batch_size=2000, shuffle=True)

    prediction_total = []
    y_data_total = []
    binary_prediction_total = []
    for data in loader:

        x_data, y_data, coordinates_samples = data
        y_data = y_data.float()

        if device.type == "cuda":
            x_data = x_data.cuda().float()
            y_data = y_data.cuda().float()
        elif device.type == "cpu":
            x_data = x_data.float()
            y_data = y_data.float()

        prediction = model(x_data)
        prediction = nn.Sigmoid()(prediction)

        prediction = torch.flatten(prediction)
        y_data = torch.flatten(y_data)

        prediction = functions.numpy_raster(prediction)
        y_data = functions.numpy_raster(y_data)

        binary_prediction = functions.convert_to_binary(prediction, 0.5)

        prediction_total.append(prediction)
        y_data_total.append(y_data)
        binary_prediction_total.append(binary_prediction)

    prediction_total = np.concatenate(prediction_total)
    y_data_total = np.concatenate(y_data_total)
    binary_prediction_total = np.concatenate(binary_prediction_total)

    return prediction_total, y_data_total, binary_prediction_total


def validation_graph(model: nn.Module, dataset: torch.utils.data.dataset, description_model: str,
                     folder_results: str) -> None:
    model = model.eval()

    from torch_geometric.data import DataLoader
    loader = DataLoader(dataset, batch_size=2000, shuffle=True)

    image_prediction, mask, binary_prediction = predict_on_graph_dataset(model, loader)

    auc = metrics.roc_auc_score(mask, image_prediction)
    report = accuracy_metrics_report(mask, binary_prediction)

    learning_rate = model.learning_rate
    results_to_save = f" learning rate= {learning_rate}\n\n" + description_model + f"\n\n auc= {auc}"
    results_to_save += f"\n\n {report}"

    file_results = f"{folder_results}results_model_{current_time}.txt"
    functions.write_text_file(path=file_results, text=results_to_save)


def predict_on_graph_dataset(model: nn.Module, dataset: torch.utils.data.dataset):
    from torch_geometric.data import DataLoader
    loader = DataLoader(dataset, batch_size=2000, shuffle=True)

    prediction_total = []
    y_data_total = []
    binary_prediction_total = []
    for data in loader:
        graph, mask, segmentation_map, coordinate = data
        graph = graph.to(device)

        segmentation_map = segmentation_map.flatten()

        with torch.no_grad():
            prediction = model.predict(graph)
        prediction = prediction.flatten()
        image_prediction = functions.graph_labels_to_image(functions.numpy_raster(prediction),
                                                           functions.numpy_raster(segmentation_map))
        image_prediction = image_prediction.flatten()
        mask = mask.flatten()

        binary_prediction = functions.convert_to_binary(image_prediction, 0.5)

        y_data_total.append(mask)
        prediction_total.append(image_prediction)
        binary_prediction_total.append(binary_prediction)

    prediction_total = np.concatenate(prediction_total)
    y_data_total = np.concatenate(y_data_total)
    binary_prediction_total = np.concatenate(binary_prediction_total)

    return prediction_total, y_data_total, binary_prediction_total


def rf_accuracy_estimation(x_train: np.array, y_train: np.array, x_test: np.array, y_test: np.array,
                           description_model: str, folder_results: str,
                           perform_cross_validation=False):
    classifier = RandomForestClassifier(max_depth=3, random_state=0, n_estimators=20)
    classifier.fit(x_train, y_train)
    prediction_label = classifier.predict(x_test)

    accuracy_metrics_report(y_test, prediction_label)

    report = accuracy_metrics_report(y_test, prediction_label)

    if perform_cross_validation:
        cross_validation_report(classifier, x_test, y_test)

    file_results = f"{folder_results}results_model_{current_time}.txt"
    functions.write_text_file(path=file_results, text=report)

    return report


def cross_validation_report(model, x_test: np.array, y_test: np.array):
    cross_validation_parameters = KFold(n_splits=10, random_state=1, shuffle=True)
    scores_cv = cross_val_score(model, x_test, y_test,
                                cv=cross_validation_parameters, scoring='f1_macro')

    print("Score cross-validation is:")
    print(scores_cv)


def accuracy_metrics_report(y_labels: np.array, y_prediction: np.array):
    report = ""

    conf_mat = confusion_matrix(y_labels, y_prediction)
    class_report = classification_report(y_labels, y_prediction)

    report += f"\n\nConfusion matrix \n\n TN - FN \n\n FP - TP \n\n {conf_mat}"
    report += f"\n\n {class_report}"

    return report
