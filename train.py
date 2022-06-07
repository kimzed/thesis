# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 12:08:25 2021

@author: baron015
"""

import torch
import torchnet as tnt

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path
from evaluate import predict_on_graph_dataset
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc as auc_scoring
import torch.nn as nn
from typing import Tuple

from sklearn import metrics
# importing our functions
import utils as fun
from evaluate import predict_on_dataset

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCELoss()
sigmoid_function = torch.nn.Sigmoid()

save_models = True

from datetime import datetime

now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

folder_cnn_save = "runs/CnnModel/"
folder_evaluation_models = f"{folder_cnn_save}evaluation_models/"

UNIT_TEST = {'is_true': False}


# to check memory allocation: torch.cuda.memory_summary(device=None, abbreviated=False)


def train_graph(model, dataset, number_epochs, description_model: str, folder_result: str):
    train_data, test_data = train_test_split(dataset, test_size=0.2)

    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model.train()
    model = model.to(device)
    # option to add weight decay, but it was not working very well
    optimizer = optim.Adam(model.parameters(), lr=model.learning_rate) #, weight_decay=1e-4)

    # slow reduction of lr
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30,50,70,80], gamma=0.1)

    metrics = {"train_loss": [], "validation_auc": [], "validation_loss": []}

    for i_epoch in range(number_epochs):

        loss_average = tnt.meter.AverageValueMeter()
        model.train()

        for batch in loader:
            if UNIT_TEST['is_true']:
                graph = batch
            else:
                graph, mask, segmentation_map, coordinate = batch

            batch = graph.to(device)
            label = graph.y.float()
            label = label[:, None]

            # ============forward===========

            prediction = model(batch)
            prediction_probability = sigmoid_function(prediction)

            # ============loss===========
            loss = criterion(prediction_probability, label)
            loss_average.add(loss.item())

            # ============backward===========
            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            scheduler.step()

        is_validation_epoch = i_epoch % 2 == 0

        if is_validation_epoch:
            validation_accuracy, loss_validation = test_graph(test_loader, model)
            print(
                "Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}. test loss {}".format(i_epoch, loss_average.mean,
                                                                                     validation_accuracy,
                                                                                     loss_validation))
            metrics["train_loss"].append(loss_average.mean)
            metrics["validation_auc"].append(validation_accuracy)
            metrics["validation_loss"].append(loss_validation)

        if save_models:
            folder_evaluation_models = "runs/Gnn/validation/"
            file_model = f"{folder_evaluation_models}save_model_epoch_{i_epoch}.pt"
            fun.save_model(model, file_model)

    # save best model
    if save_models:
        fun.saving_best_model(model, folder_evaluation_models, metrics, description_model, folder_result)

        fun.delete_files_from_folder(folder_evaluation_models)

    file_plot = f"{folder_result}plot_model_{current_time}.png"
    fun.visualisation_losses(metrics, file_plot)

    return model


def test_graph(loader, model) -> tuple:
    model.eval()

    loss_average = tnt.meter.AverageValueMeter()

    total_accuracy = 0

    number_samples = len(loader)

    # TODO selection per auc, decide what we take
    if not UNIT_TEST['is_true']:
        image_prediction, mask, binary_prediction = predict_on_graph_dataset(model, loader)
        precision, recall, thresholds = precision_recall_curve(mask, image_prediction)
        auc = auc_scoring(recall, precision)
        loss = criterion(torch.tensor(image_prediction), torch.tensor(mask))
        loss_average.add(loss.item())
        average_loss = loss_average.mean

        return auc, average_loss

    for data in loader:
        if UNIT_TEST['is_true']:
            graph = data
        else:
            graph, mask, segmentation_map, coordinate = data
            segmentation_map = segmentation_map.flatten()
        graph = graph.to(device)

        with torch.no_grad():
            graph_for_binary = graph.clone()
            prediction = model(graph)
            prediction_probability = sigmoid_function(prediction)
            if device.type == "cuda":
                label = graph.y.float().cuda()
                label = label[:, None]
            else:
                label = graph.y.float()
                label = label[:, None]
        loss = criterion(prediction_probability, label)

        loss_average.add(loss.item())
        prediction_binary = model.predict_to_binary(graph_for_binary)
        if UNIT_TEST['is_true']:
            prediction = fun.numpy_raster(prediction_binary)
            label = fun.numpy_raster(label)
            accuracy = accuracy_score(prediction, label, normalize=True)
            total_accuracy += accuracy


        else:
            prediction = prediction_binary.flatten()
            image_prediction = fun.graph_labels_to_image(fun.numpy_raster(prediction),
                                                         fun.numpy_raster(segmentation_map))
            image_prediction = image_prediction.flatten()
            mask = mask.flatten()
            accuracy = accuracy_score(image_prediction, mask, normalize=True)
            total_accuracy += accuracy

    # total_number_nodes = fun.get_number_nodes(loader.dataset)

    total_accuracy = total_accuracy / number_samples

    average_loss = loss_average.mean

    return total_accuracy, average_loss

def test_cnn(model:nn.Module, data_loader) -> float:
    model.eval()

    prediction, y_data, binary_prediction = predict_on_dataset(model, data_loader)

    precision, recall, thresholds = precision_recall_curve(y_data, prediction)
    auc = auc_scoring(recall, precision)

    loss_average = tnt.meter.AverageValueMeter()
    loss = criterion(torch.tensor(prediction), torch.tensor(y_data))
    loss_average.add(loss.item())
    average_loss = loss_average.mean

    return average_loss, auc

def train_one_epoch(model, data_loader):
    model.train()
    model.to(device)

    scheduler = torch.optim.lr_scheduler.MultiStepLR(model.optimizer, milestones=[30, 50, 70, 80], gamma=0.1)

    loss_average = tnt.meter.AverageValueMeter()
    accuracy_metric = 0.0

    for i_batch, data in enumerate(data_loader):

        x_data, y_data, coordinates_samples = data
        positives_in_y_data = 1 in y_data

        if device.type == "cuda":
            x_data = x_data.cuda().float()
            y_data = y_data.cuda().float()
        elif device.type == "cpu":
            x_data = x_data.float()
            y_data = y_data.float()

        # ============forward===========

        prediction = model(x_data)

        # ============loss===========

        prediction_flattened = torch.flatten(prediction, start_dim=1)
        probability_prediction = sigmoid_function(prediction_flattened)

        if positives_in_y_data:
            indexes_positives = y_data == 1
            indexes_negatives = y_data == 0
            loss_positives = criterion(probability_prediction[indexes_positives], y_data[indexes_positives])
            loss_negatives = criterion(probability_prediction[indexes_negatives], y_data[indexes_negatives])
            loss = loss_positives + loss_negatives
        else:
            loss = criterion(probability_prediction, y_data)

        loss_average.add(loss.item())

        accuracy_metric += fun.get_accuracy(prediction_flattened, y_data)

        # ============backward===========

        model.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        model.optimizer.step()
        scheduler.step()

        torch.cuda.empty_cache()

    number_batches = i_batch + 1
    accuracy_metric_averaged = accuracy_metric / number_batches

    return loss_average.value()[0], accuracy_metric_averaged


def train(model, dataset, number_epochs: int, description_model: str, results_folder: Path):
    import dataset as dataset_functions
    from torch.utils.data import DataLoader

    train_data, validation_data = dataset_functions.split_dataset(dataset, test_size=0.8)
    train_loader = DataLoader(train_data, batch_size=32
                              , shuffle=True)
    #validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

    fun.show_characteristics_model(model)

    losses = {"train_loss": [], "train_acuracy": [], "validation_auc": [], "validation_loss": []}

    for i_epoch in range(number_epochs):

        loss_train, accuracy = train_one_epoch(model, train_loader)

        is_validation_epoch = i_epoch % 2 == 0

        if is_validation_epoch:
            #loss_val, accu_val = train_one_epoch(model, validation_loader, validation=True)
            loss_val, auc_val = test_cnn(model, validation_data)

        losses["train_loss"].append(loss_train)
        losses["train_acuracy"].append(accuracy)
        losses["validation_loss"].append(loss_val)
        losses["validation_auc"].append(auc_val)

        fun.display_training_metrics(i_epoch, loss_train, accuracy)

        if save_models:
            file_model = f"{folder_evaluation_models}save_model_epoch_{i_epoch}.pt"
            fun.save_model(model, file_model)

    if save_models:
        print("Saving best model")
        i_epoch_best = fun.saving_best_model(model, folder_evaluation_models, losses, description_model, results_folder)

        # cleaning the evaluation models saved every epoch
        fun.delete_files_from_folder(folder_evaluation_models)

    file_plot = f"{results_folder}plot_model_{current_time}_epoch.png"
    fun.visualisation_losses(losses, file_plot)

    model.eval()

    return model
