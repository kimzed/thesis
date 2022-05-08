import evaluate
from evaluate import predict_on_dataset
import dataset
from torch.utils.data import ConcatDataset
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
folder_result = "python/unit_test/fake_result_runs/"


def test_predict_on_dataset():
    import model.cnn as CNN
    model = CNN.CnnSemanticSegmentation()
    model.to(device)

    folders_data, folders_labels = dataset.get_data_folders([2019])
    datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)
    dataset_full = ConcatDataset(datasets)

    prediction_total, y_data_total, binary_prediction_total = predict_on_dataset(model, dataset_full)

    assert (prediction_total.shape == y_data_total.shape == binary_prediction_total.shape)
    assert (np.all(0 <= binary_prediction_total))
    assert (np.all(binary_prediction_total <= 1))
    assert (np.all(0 <= y_data_total))
    assert (np.all(y_data_total <= 1))
    assert (np.all(0 <= prediction_total))
    assert (np.all(prediction_total <= 1))


def test_rf_accuracy_estimation_results_on_dummy_dataset_are_correct():
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    iris = datasets.load_iris()
    import pandas as pd
    data = pd.DataFrame({
        'sepal length': iris.data[:, 0],
        'sepal width': iris.data[:, 1],
        'petal length': iris.data[:, 2],
        'petal width': iris.data[:, 3],
        'species': iris.target
    })
    X = data[['sepal length', 'sepal width', 'petal length', 'petal width']]  # Features
    y = data['species']  # Labels
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    report = evaluate.rf_accuracy_estimation(x_train, y_train, x_test, y_test, folder_results=folder_result,
                                    description_model="unit_test")

    print(report)

test_predict_on_dataset()
test_rf_accuracy_estimation_results_on_dummy_dataset_are_correct()
