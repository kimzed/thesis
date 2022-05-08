import dataset
from torch.utils.data import ConcatDataset
import numpy as np


def test_get_x_and_y_from_subset_dataset_first_sample_is_correct():
    folders_data, folders_labels = dataset.get_data_folders([2019])
    datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)
    dataset_full = ConcatDataset(datasets)
    _, test_dataset = dataset.split_dataset_geographically(dataset_full, x_limit=36.523466)

    expected_x = test_dataset.__getitem__(0)[0]
    expected_y = test_dataset.__getitem__(0)[1]

    expected_x = expected_x.reshape(expected_x.shape[0], expected_x.shape[1] ** 2)
    expected_x = np.moveaxis(expected_x, 0, -1)

    number_pixels_image = expected_x.shape[0]
    number_pixels_label = expected_y.shape[0]

    x, y = dataset.get_x_and_y_from_subset_dataset(test_dataset)

    actual_x = x[:number_pixels_image]
    actual_y = y[:number_pixels_label]

    assert (np.array_equal(expected_x, actual_x))
    assert (np.array_equal(expected_y, actual_y))


def test_get_x_and_y_from_subset_dataset_second_sample_is_correct():
    folders_data, folders_labels = dataset.get_data_folders([2019])
    datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)
    dataset_full = ConcatDataset(datasets)
    _, test_dataset = dataset.split_dataset_geographically(dataset_full, x_limit=36.523466)

    expected_x = test_dataset.__getitem__(1)[0]
    expected_y = test_dataset.__getitem__(1)[1]

    expected_x = expected_x.reshape(expected_x.shape[0], expected_x.shape[1] ** 2)
    expected_x = np.moveaxis(expected_x, 0, -1)

    number_pixels_image = expected_x.shape[0]

    x, y = dataset.get_x_and_y_from_subset_dataset(test_dataset)

    actual_x = x[number_pixels_image:number_pixels_image*2]
    actual_y = y[number_pixels_image:number_pixels_image*2]

    assert (np.array_equal(expected_x, actual_x))
    assert (np.array_equal(expected_y, actual_y))

def test_get_x_and_y_from_subset_dataset_last_sample_is_correct():
    folders_data, folders_labels = dataset.get_data_folders([2019])
    datasets = dataset.get_datasets(data_folders=folders_data, label_folders=folders_labels)
    dataset_full = ConcatDataset(datasets)
    _, test_dataset = dataset.split_dataset_geographically(dataset_full, x_limit=36.523466)

    expected_x = test_dataset.__getitem__(-1)[0]
    expected_y = test_dataset.__getitem__(-1)[1]

    expected_x = expected_x.reshape(expected_x.shape[0], expected_x.shape[1] ** 2)
    expected_x = np.moveaxis(expected_x, 0, -1)

    number_pixels_image = expected_x.shape[0]

    x, y = dataset.get_x_and_y_from_subset_dataset(test_dataset)

    actual_x = x[-number_pixels_image:]
    actual_y = y[-number_pixels_image:]

    assert (np.array_equal(expected_x, actual_x))
    assert (np.array_equal(expected_y, actual_y))


test_get_x_and_y_from_subset_dataset_first_sample_is_correct()
test_get_x_and_y_from_subset_dataset_second_sample_is_correct()
test_get_x_and_y_from_subset_dataset_last_sample_is_correct()
