
from pathlib import Path
import numpy as np
import os

from utils import graph_labels_to_image
from pre_processing.pre_processing_graph_data import get_most_frequent_int_value, get_node_label

TEST_DATA_FOLDER = Path(__file__).parent.parent.parent.joinpath("test").resolve()

def test_get_node_label_last_superpixels_has_correct_label():

    superpixels = np.load(os.path.join(TEST_DATA_FOLDER, "superpixels.npy"))
    node_label_image = np.load(os.path.join(TEST_DATA_FOLDER, "node_labels_image.npy"))

    segmentation_mask = np.load(os.path.join(TEST_DATA_FOLDER, 'segmentation_mask.npy'))

    excepted_label = 1

    max_label = np.max(superpixels)

    computed_labels = get_node_label(segmentation_mask, superpixels)
    computed_label = computed_labels[-1]

    assert (computed_label == excepted_label)

test_get_node_label_last_superpixels_has_correct_label()
