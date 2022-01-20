# -*- coding: utf-8 -*-
"""
Created on Sun Jul 25 11:55:21 2021

@author: baron015
"""

# loading required packages
import os
from scipy.interpolate import RegularGridInterpolator
import random
from scipy import interpolate
from random import random as randnb
import numpy as np
import matplotlib.pyplot as plt
from random import randint
from sklearn import metrics



# libraries from GNN environment
import torch_geometric
import torch.nn as nn
import torch




## changing working directory
os.chdir("//WURNET.NL/Homes/baron015/My Documents/thesis")


def get_number_nodes(graphs: list):

    total_number_nodes = 0

    for graph in graphs:
        number_nodes = len(graph.y)
        total_number_nodes += number_nodes

    return total_number_nodes

def transform_numpy(img, mask, p=0.5):
    """
    Function to add transformation on numpy matrixs
    """
    
    mask = mask.reshape(38, 38)
    img = np.moveaxis(img, 0, 2)
    
    if randnb() > p:
        img = np.fliplr(img)
        mask = np.fliplr(mask)
        
    if randnb() > p:
        img = np.flipud(img)
        mask = np.flipud(mask)
    
    img = np.moveaxis(img, 2, 0)
    
    
    return  img, mask.flatten()


def get_accuracy(prediction, y_data, validation=False):


    binary_prediction = convert_to_binary(numpy_raster(prediction), 0.5)
    binary_prediction.mean()
    y_data.mean()

    if validation:
        accuracy = metrics.f1_score(y_true=numpy_raster(y_data).flatten(), y_pred=binary_prediction.flatten(), average="macro")
    else:
        accuracy = metrics.accuracy_score(y_true=numpy_raster(y_data).flatten(), y_pred=binary_prediction.flatten())




    return accuracy


def convert_to_binary(values, threshold):

    data = np.array(values)

    data_is_above_threshold = data > threshold

    binary_vector = np.zeros(data.shape)
    binary_vector[data_is_above_threshold] = 1

    return binary_vector


def show_characteristics_model(model):

    print('Total number of encoder parameters: {}'.format(sum([parameter.numel() for parameter in model.parameters()])))
    print(model)


def display_training_metrics(i_epoch: int, loss_train, accuracy):

    print('Epoch %3d -> Train Loss: %1.4f' % (i_epoch, loss_train))
    print("f1 score is %1.4f" % (accuracy))
    print("\n")

def visualize_sample(dataset, idx):
    """
    param: a raster 2*128*128, with dem and radiometry
    fun: visualize a given raster in two dimensions and in 3d for altitude
    """
    
    img, mask = dataset.__getitem__(idx)

    nb_px = img.shape[-1]

    # resize the labels
    mask = mask.reshape(nb_px, nb_px)

    # select only rgb bands and normalize
    img = img[[2,1,0],:,:]
    if np.any(img < 0):
        img = img + np.absolute(img.min())
    img *= (1/img.max())
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)

    # creating axes and figures
    fig, (rgb, lab) = plt.subplots(1, 2, figsize=(14, 14)) # Create one plot with figure size 10 by 10

    # setting the title
    rgb.set_title("image")
    lab.set_title("mask")
    rgb.axis("off")
    lab.axis("off")

    # showing the data
    rgb = rgb.imshow(img)#, vmin=100, vmax=1000)
    lab = lab.imshow(mask)

    plt.show()
    

def visualize_landsat_array(landsat_array: np.ndarray):
    
    
    img = landsat_array[[3,2,1],:,:]
    if np.any(img < 0):
        img = img + np.absolute(img.min())
    img *= (1/img.max())
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    
    plt.imshow(img)
    plt.show()
    
    
def visualize_prediction(dataset, model, idx):
    """
    param: a raster 2*128*128, with dem and radiometry
    fun: visualize a given raster in two dimensions and in 3d for altitude
    """
    
    img, mask = dataset.__getitem__(idx)
    
    # adding one dimension (batch)
    prediction = nn.Sigmoid()(model((torch_raster(img)[None,:,:,:])))
    
    nb_px = img.shape[-1]
    
    # resize the labels
    mask = mask.reshape(nb_px, nb_px)
    
    # select only rgb bands and normalize
    img = img[[3,2,1],:,:]
    img *= (1/img.max())
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 0, 1)
    
    
    
    # creating axes and figures
    fig, (rgb, lab, pred) = plt.subplots(1, 3, figsize=(14, 14)) # Create one plot with figure size 10 by 10
    
    # setting the title
    rgb.set_title("image")
    lab.set_title("mask")
    pred.set_title("prediction")
    rgb.axis("off")
    lab.axis("off")
    pred.axis("off")
    
    # showing the data
    rgb = rgb.imshow(img)#, vmin=100, vmax=1000)
    lab = lab.imshow(mask)
    pred = pred.imshow(numpy_raster(prediction), vmin=0, vmax=1)
    
    plt.show()

def interpolate_missing_pixels(
        image: np.ndarray,
        mask: np.ndarray,
        method: str = 'nearest',
        fill_value: int = 0):
    """
    :param image: a 2D image
    :param mask: a 2D boolean image, True indicates missing values
    :param method: interpolation method, one of
        'nearest', 'linear', 'cubic'.
    :param fill_value: which value to use for filling up data outside the
        convex hull of known pixel values.
        Default is 0, Has no effect for 'nearest'.
    :return: the image with missing values interpolated
    """

    h, w = image.shape[:2]
    xx, yy = np.meshgrid(np.arange(w), np.arange(h))

    known_x = xx[~mask]
    known_y = yy[~mask]
    known_v = image[~mask]
    missing_x = xx[mask]
    missing_y = yy[mask]

    interp_values = interpolate.griddata(
        (known_x, known_y), known_v, (missing_x, missing_y),
        method=method, fill_value=fill_value
    )

    interp_image = image.copy()
    interp_image[missing_y, missing_x] = interp_values

    return interp_image


def graph_labels_to_image(labels: np.array, superpixel_map: np.array):

    dimensions_image = superpixel_map.shape
    image_label = np.zeros(dimensions_image)
    number_nodes = superpixel_map.max()

    for i_node in range(number_nodes):

        label_node = labels[i_node]
        i_image_node = superpixel_map == i_node
        image_label[i_image_node] = label_node

    return image_label

def graph_files_to_node_data(graphs:list):
    node_features = [graph_to_node_features(graph_file) for graph_file in graphs]
    labels = [graph_to_labels(graph_file) for graph_file in graphs]
    x = np.concatenate(node_features)
    y = np.concatenate(labels)

    return x, y


def normalize_nodes_features(graph):
    mean = graph.x.mean()
    std = graph.x.std()
    graph.x = (graph.x - mean) / std

    return graph


def balancing_binary_dataset(x: np.array, y: np.array):

    idxs_ones = y == 1
    idx_zeros = y == 0

    data_ones, label_ones = x[idxs_ones.flatten(),], y[idxs_ones.flatten(),]
    data_zeros, label_zeros = x[idx_zeros.flatten(),], y[idx_zeros.flatten(),]

    number_ones = data_ones.shape[0]
    numner_zeros = data_zeros.shape[0]

    number_samples_weaker_class = min(number_ones, numner_zeros)

    data_zeros, label_zeros = data_zeros[:number_samples_weaker_class+50,], label_zeros[:number_samples_weaker_class+50,]

    x_balanced = np.concatenate([data_zeros, data_ones])
    y_balanced = np.concatenate([label_zeros, label_ones])

    return x_balanced, y_balanced

def converting_probability_array_to_binary(array, threshold:float):

    dimensions_array = array.shape
    binary_result = torch.ones(dimensions_array)
    values_below_threshold = array < threshold
    binary_result[values_below_threshold] = 0

    return binary_result


def graph_to_node_features(graph_file: str)-> np.array:
    graph = torch.load(graph_file)
    features = graph.x
    features = np.array(features)

    return features

def graph_to_labels(graph_file: str) -> np.array:
    graph = torch.load(graph_file)
    labels = graph.y
    labels = np.array(labels)

    # we add one dimension to allow concatenation for multiple samples
    labels = labels[:, None]

    return labels


def count_value(array: np.array, value):

    array.flatten()
    out = np.count_nonzero(array == value)

    return out


def set_seed(seed, cuda=True):
        """ 
        Sets seeds
        """
        
        # setting the seed for various libraries
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if cuda: 
            torch.cuda.manual_seed(seed)  


def visualisation_losses(losses: dict):
    
    plt.title('Metrics per number of epochs')
    plt.xlabel('epoch')
    plt.ylabel("metric")

    for name_loss in losses:

        list_epochs = range(len(losses[name_loss]))
        plt.plot(list_epochs, losses[name_loss], label=name_loss)
        
    plt.legend()
    plt.show()


def torch_raster(raster, cuda=False):
    """
    function that adapts a raster for the model, change to torch tensor, on cuda,
    float
    """
    
    # converting the data
    if cuda:
        result = torch.from_numpy(raster).cuda().float()
    else:
        result = torch.from_numpy(raster).float()
        
    return result


def z_score(array: np.array):
    
    mu = array.mean()
    std = array.std()
    
    normalized_array = (array - mu) / std
    
    return normalized_array

def numpy_raster(raster):
    """
    function that adapts a raster for the model, change to torch tensor, on cuda,
    float
    """
    
    # converting the result
    result = raster.detach().cpu().numpy().squeeze()
    
    return result


def regrid(data, out_x, out_y, interp_method="linear"):
    """
    param: numpy array, number of coludem, number of rows
    fun: function to interpolate a raster
    
    """
    
    m = max(data.shape[-2], data.shape[-1])
    y = np.linspace(0, 1.0/m, data.shape[-2])
    x = np.linspace(0, 1.0/m, data.shape[-1])
    interpolating_function = RegularGridInterpolator((y, x), data, method=interp_method)
    yv, xv = np.meshgrid(np.linspace(0, 1.0/m, out_y), np.linspace(0, 1.0/m, out_x))
    
    # reprojects the data
    return interpolating_function((xv, yv))




def save_numpy_data(arrays: list, path, file_name):
    

    for i_file, array in enumerate(arrays):
        
        total_file_path = f"{path}{file_name}{i_file}.npy"
        np.save(total_file_path, array)
    
def put_bands_in_last_dimension(raster: np.array):
    
    raster = np.swapaxes(raster, 1, 0)
    raster = np.swapaxes(raster, 1, 2)
    
    return raster
    


def get_files(dir_files):
    """
    Get all the files from the dir in a list with the complete file path
    """
    
    # empty list to store the files
    list_files = []
    
    # getting the files in the list
    for path in os.listdir(dir_files):
        full_path = os.path.join(dir_files, path)
        if os.path.isfile(full_path):
            list_files.append(full_path)
            
    return list_files



def select_random_item(my_list: list):

    len_list = len(my_list)
    idx = randint(0, len_list)
    item = my_list[idx]

    return item


def resize_labels_into_image(labels, image):

    dimension_image = image.shape[0]
    image_label =  labels.reshape((dimension_image, dimension_image))

    return image_label


def create_directory(path: str):
    
    try:
        os.makedirs(path)
    except OSError:
        print ("Creation of the directory %s failed" % path)
    else:
        print ("Successfully created the directory %s" % path)


def data_folder_has_positives(data_folder: str) -> bool:
    return "greenhouse" in data_folder

def save_model(model, path:str, description:str):
    path_log_message = f"{path}.txt"
    torch.save(model.state_dict(), path)
    write_text_file(path_log_message, description)
    write_text_file(path_log_message, str(model))

def write_text_file(path: str, text: str):
    f = open(path, "w")
    f.write(text)
    f.close()


def load_weights_to_model(model: nn.Module, path_weights: str):

    model.load_state_dict(torch.load(path_weights))
    model.eval()

    return model


def delete_files_from_folder(path_folder):
    list_files = get_files(path_folder)
    [os.remove(file) for file in list_files]


def read_txt_file_into_string(path):
    with open(path, "r") as myfile:
        data = myfile.read().replace('\n', ' ')
    return data




