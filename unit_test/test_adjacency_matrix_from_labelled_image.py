from pre_processing.pre_processing_graph_data import adjacency_matrix_from_labelled_image
import numpy as np

def test_adjacency_matrix_from_labelled_image_fully_connected_matrix_is_correct():

    fully_connected = True
    fake_segmentation_mask = np.array([[0,0,0],
                                       [0,1,1],
                                       [2,2,2]])

    fully_connected_adjacency_matrix = adjacency_matrix_from_labelled_image(fake_segmentation_mask, fully_connected=fully_connected)

    assert (fully_connected_adjacency_matrix.shape[0] == fake_segmentation_mask.max()+1)


test_adjacency_matrix_from_labelled_image_fully_connected_matrix_is_correct()