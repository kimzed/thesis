import dataset_graphs

def test_get_graphs_from_subset_dataset_outputs_correct_number_samples():

    folder_graphs, folders_labels, folder_semantic_maps = dataset_graphs.get_data_folders([2019])
    dataset_full = dataset_graphs.merge_datasets(folders_labels, folder_graphs, folder_semantic_maps)

    _, test_dataset = dataset_graphs.split_dataset_geographically(dataset_full, x_limit=36.523466)

    number_samples = len(test_dataset)


    graph_files = dataset_graphs.get_graphs_from_subset_dataset(test_dataset)

    number_files = len(graph_files)

    assert (number_samples == number_files)

test_get_graphs_from_subset_dataset_outputs_correct_number_samples()