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

# importing our functions
import utils as fun

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


# to check memory allocation: torch.cuda.memory_summary(device=None, abbreviated=False)


def train_graph(model, dataset, number_epochs, description_model: str, folder_result: str):
    train_data, test_data= train_test_split(dataset, test_size=0.2)

    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model.train()
    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=model.learning_rate)

    metrics = {"train_loss": [], "validation_accuracy": [], "validation_loss": []}

    for i_epoch in range(number_epochs):

        loss_average = tnt.meter.AverageValueMeter()
        model.train()

        for batch in loader:
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
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        is_validation_epoch = i_epoch % 2 == 0

        if is_validation_epoch:
            validation_accuracy, loss_validation = test_graph(test_loader, model)
            print(
                "Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(i_epoch, loss_average.mean, validation_accuracy))
            metrics["train_loss"].append(loss_average.mean)
            metrics["validation_accuracy"].append(validation_accuracy)
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

    for data in loader:

        graph, mask, segmentation_map, coordinate = data
        graph = graph.to(device)

        segmentation_map = segmentation_map.flatten()

        with torch.no_grad():
            prediction = model.predict(graph)
            if device.type == "cuda":
                label = graph.y.float().cuda()
                label = label[:, None]
            else:
                label = graph.y.float()
                label = label[:, None]
        loss = criterion(prediction, label)

        loss_average.add(loss.item())

        prediction = prediction.flatten()
        image_prediction = fun.graph_labels_to_image(fun.numpy_raster(prediction), fun.numpy_raster(segmentation_map))
        image_prediction = image_prediction.flatten()
        mask = mask.flatten()
        accuracy = accuracy_score(image_prediction, mask, normalize=True)
        total_accuracy += accuracy

    # total_number_nodes = fun.get_number_nodes(loader.dataset)

    total_accuracy = total_accuracy / number_samples

    average_loss = loss_average.mean

    return total_accuracy, average_loss


def train_one_epoch(model, data_loader, validation=False):
    model.train()
    model.to(device)

    if validation:
        model.eval()

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

        accuracy_metric += fun.get_accuracy(prediction_flattened, y_data, validation)

        # ============backward===========

        model.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if not validation:
            model.optimizer.step()

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
    validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True)

    fun.show_characteristics_model(model)

    losses = {"train_loss": [], "train_acuracy": [], "validation_loss": [], "validation_accuracy": []}

    for i_epoch in range(number_epochs):

        loss_train, accuracy = train_one_epoch(model, train_loader)

        is_validation_epoch = i_epoch % 2 == 0

        if is_validation_epoch:
            loss_val, accu_val = train_one_epoch(model, validation_loader, validation=True)

        losses["train_loss"].append(loss_train)
        losses["train_acuracy"].append(accuracy)
        losses["validation_loss"].append(loss_val)
        losses["validation_accuracy"].append(accu_val)

        fun.display_training_metrics(i_epoch, loss_train, accuracy)

        if save_models:
            file_model = f"{folder_evaluation_models}save_model_epoch_{i_epoch}.pt"
            fun.save_model(model, file_model)

    if save_models:
        print("Saving best model")
        fun.saving_best_model(model, folder_evaluation_models, losses, description_model, results_folder)

        # cleaning the evaluation models saved every epoch
        fun.delete_files_from_folder(folder_evaluation_models)

    file_plot = f"{results_folder}plot_model_{current_time}.png"
    fun.visualisation_losses(losses, file_plot)

    model.eval()

    return model
