

import torch
import torchnet as tnt

from torch_geometric.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# importing our functions
import utils as fun
import torch.nn.functional as F

import torch.optim as optim

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

sigmoid_function = torch.nn.Sigmoid()

save_models = False

from datetime import datetime
now = datetime.now()
current_time = now.strftime("%Y_%m_%d_%H_%M")

folder_cnn_save = "runs/CnnModel/"
folder_evaluation_models = f"{folder_cnn_save}evaluation_models/"



def train_graph(model, dataset, number_epochs, description_model:str):

    test_data, train_data = train_test_split(dataset, test_size=0.8)

    loader = DataLoader(train_data, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=True)

    model = model.to(device)
    opt = optim.Adam(model.parameters(), lr=0.002)

    metrics = {"train_loss": [],"validation_loss":[]}

    for i_epoch in range(number_epochs):

        loss_average = tnt.meter.AverageValueMeter()
        model.train()

        for graph in loader:


            batch = graph.to(device)
            label = graph.y
            #label = label[:,None]

            # ============forward===========
            prediction = model(batch)

            # ============loss===========
            loss =  F.cross_entropy(prediction, label)
            loss_average.add(loss.item())

            # ============backward===========
            opt.zero_grad()
            loss.backward(retain_graph=True)
            opt.step()

        is_validation_epoch = i_epoch % 2 == 0

        if is_validation_epoch:

            loss_validation = test_graph(test_loader, model)
            print("Epoch {}. Training loss is {:.4f} Validation loss: {:.4f}".format(i_epoch, loss_average.mean, loss_validation))
            metrics["train_loss"].append(loss_average.mean)
            metrics["validation_loss"].append(loss_validation)


        if save_models:
            folder_evaluation_models = "runs/Gnn/validation/"
            file_model = f"{folder_evaluation_models}save_model_epoch_{i_epoch}.pt"
            fun.save_model(model, file_model)

    # save best model
    if save_models:
        fun.saving_best_model(model, folder_evaluation_models, metrics, description_model)

        fun.delete_files_from_folder(folder_evaluation_models)

        folder_save_plot = fun.jump_up_folders(folder_evaluation_models, 1)
        file_plot = f"{folder_save_plot}plot_model_{current_time}.png"
        fun.visualisation_losses(metrics, file_plot)

    return model

def test_graph(loader, model)-> tuple:
    model.eval()

    loss_average = tnt.meter.AverageValueMeter()

    for graph in loader:

        graph = graph.to(device)

        with torch.no_grad():
            prediction = model(graph)
            label = graph.y
        loss = F.cross_entropy(prediction, label)
        loss_average.add(loss.item())

    average_loss = loss_average.mean

    return average_loss
