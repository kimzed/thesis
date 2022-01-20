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


# importing our functions
import utils as fun

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = torch.nn.BCELoss()
sigmoid_function = torch.nn.Sigmoid()

save_models = True

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H:%M")


def train_graph(model, dataset, number_epochs):

    # TODO dont use dataloader for train test individual graphs to test real accuracy (with segmentation map)
    test_data, train_data = train_test_split(dataset, test_size=0.8)
    loader = torch.utils.data.DataLoader(train_data, batch_size=2, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=2, shuffle=True)

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(number_epochs):

        loss_average = tnt.meter.AverageValueMeter()
        model.train()

        for batch in loader:


            batch = batch.to(device)
            label = batch.y.float()
            label = label[:,None]

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

        if epoch % 10 == 0:

            test_acc = test_graph(test_loader, model)
            print("Epoch {}. Loss: {:.4f}. Test accuracy: {:.4f}".format(epoch, loss_average.mean, test_acc))

    return model


def test_graph(loader, model):
    model.eval()

    correct = 0
    for data in loader:
        data = data.to(device)

        with torch.no_grad():
            pred = model.predict(data)
            label = data.y

        count_correct = accuracy_score(label, pred, normalize=False)
        correct += count_correct

    total_number_nodes = fun.get_number_nodes(loader.dataset)

    total_accuracy = correct / total_number_nodes

    return total_accuracy


def train_one_epoch(model, data_loader, validation=False):

    model.train()

    if validation:
        model.eval()

    loss_average = tnt.meter.AverageValueMeter()
    accuracy_metric = 0.0

    for i_batch, data in enumerate(data_loader):

        x_data, y_data = data
        y_data = y_data.float()
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
            loss = loss_positives*10 + loss_negatives*0.1
        else:
            loss = criterion(probability_prediction, y_data)

        loss_average.add(loss.item())


        accuracy_metric += fun.get_accuracy(prediction_flattened, y_data, validation)

        # ============backward===========

        model.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        if not validation:
            model.optimizer.step()

    number_batches = i_batch + 1
    accuracy_metric_averaged = accuracy_metric / number_batches

    return loss_average.value()[0], accuracy_metric_averaged


def train(model, dataset, number_epochs:int, description_model: str):

    import dataset as dataset_functions
    from torch.utils.data import DataLoader
    number_samples = dataset.__len__()
    train_data, validation_data = dataset_functions.split_dataset(dataset, test_size=0.8)
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    validation_loader = DataLoader(validation_data, batch_size=number_samples, shuffle=True)

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

        folder_cnn_save = "runs/CnnModel/"

        if save_models:
            path_model = f"{folder_cnn_save}save_model_epoch_{i_epoch}.pt"
            fun.save_model(model, path_model, description_model)


    if save_models:
        print("saving best model")
        best_accuracy = max(losses["validation_accuracy"])
        i_best_model = losses["validation_accuracy"].index(best_accuracy)


        path_best_model = f"{folder_evaluation_models}save_model_epoch_{i_best_model}.pt"
        path_log_message_best_model = f"{path_best_model}.txt"
        description_model = fun.read_txt_file_into_string(path_log_message_best_model)

        model = fun.load_weights_to_model(model, path_best_model)
        path_best_model = f"{folder_cnn_save}/cnn_{current_time}.pt"
        fun.save_model(model, path_best_model, description_model)

        # cleaning the evaluation models saved every epoch
        fun.delete_files_from_folder(folder_evaluation_models)


    fun.visualisation_losses(losses)

    model.eval()

    return model

