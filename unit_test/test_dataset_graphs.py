
import dataset_graphs

years = [2019, 2014, 2011]

def test_get_data_folders_get_positive_samples_is_working():
    folder_graphs, folders_labels, folder_semantic_maps = dataset_graphs.get_data_folders(years, rasters_with_positives_only=True)
    dataset_full = dataset_graphs.merge_datasets(folders_labels, folder_graphs, folder_semantic_maps)

    number_samples_expected = 607
    assert (len(dataset_full) == number_samples_expected)

test_get_data_folders_get_positive_samples_is_working()