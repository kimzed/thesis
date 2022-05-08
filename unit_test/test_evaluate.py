from evaluate import predict_on_dataset
import dataset
from torch.utils.data import ConcatDataset
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_predict_on_dataset():

    import model.cnn as CNN
    model = CNN.CnnSemanticSegmentation()
    model.to(device)

    folders_data, folders_labels = dataset.get_data_folders([2019])
    datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)
    dataset_full = ConcatDataset(datasets)

    prediction_total, y_data_total, binary_prediction_total = predict_on_dataset(model, dataset_full)

    assert (prediction_total.shape == y_data_total.shape == binary_prediction_total.shape)
    assert (np.all( 0 <= binary_prediction_total ))
    assert (np.all(binary_prediction_total <= 1))
    assert (np.all(0 <= y_data_total))
    assert (np.all(y_data_total <= 1))
    assert (np.all(0 <= prediction_total))
    assert (np.all(prediction_total <= 1))



test_predict_on_dataset()