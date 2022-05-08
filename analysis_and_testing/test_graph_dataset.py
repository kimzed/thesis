

import os

working_directory = "C:/Users/57834/Documents/thesis/"
os.chdir(working_directory)


import dataset_graphs as dataset
import utils as functions
import matplotlib.pyplot as plt
from random import randint
import numpy as np
import torch
from skimage.segmentation import slic

def main():

    print("loading the data")
    year = 2011
    folder_greenhouse_dataset = f"data/{year}/greenhouse_dataset/"

    folder_graphs = os.path.join(folder_greenhouse_dataset, "graphs")
    folder_semantic_maps = os.path.join(folder_greenhouse_dataset, "semantic_maps_graphs")
    folder_labels = os.path.join(folder_greenhouse_dataset, "ground_truth_rasters")
    folder_imgs = os.path.join(folder_greenhouse_dataset, "landsat_rasters")

    print("visualization...")

    import pre_processing.pre_processing_graph_data as graph_functions

    graph_functions.visualize_random_edge_index(folder_graphs, folder_semantic_maps, folder_labels, folder_imgs)

    graph_functions.visualize_graph_sample_from_data_folders(folder_graphs, folder_semantic_maps, folder_labels, folder_imgs)


    dataset_graphs = dataset.GraphDatasetSemanticSegmentation(folder_graphs, folder_labels, folder_semantic_maps)

    semantic_maps = dataset_graphs.segmentation_map_files

    ex_semantic_map = np.load(semantic_maps[0])

    """
    ### test on slic

    folder_greenhouse_dataset = "data/greenhouse_dataset/"

    folder_gt = "ground_truth_rasters/"
    folder_landsat = "landsat_rasters/"

    folder_semantic_maps = "semantic_maps_graphs/"
    folder_graphs = "graphs/"

    files_gt = functions.get_files(folder_greenhouse_dataset + folder_gt)
    files_landsat = functions.get_files(folder_greenhouse_dataset + folder_landsat)

    len_dataset = len(files_landsat)
    i_sample = randint(0, len_dataset)
    image = files_landsat[i_sample]
    gt = files_gt[i_sample]


    array_image = np.load(image)
    array_gt = np.load(gt)
    functions.visualize_landsat_array(array_image)
    plt.imshow(array_gt.reshape((38, 38)))
    plt.show()

    array_slic = functions.put_bands_in_last_dimension(array_image)
# the segments also become more rectangular and grid like.
    mask_segmentation_3bands = slic(array_slic[:,:,:3], n_segments=100, compactness=0.001, sigma=0.01,
                             multichannel=True, convert2lab=True)
    incr = 0

    idxs = np.arange(0,5,0.1)
    i = idxs[incr]
    print(f"sigma = {i}")
    mask_segmentation = slic(array_slic, n_segments=60, compactness=0.001, sigma=i,
                             multichannel=True)
    incr +=1
    plt.imshow(mask_segmentation)
    plt.show()

    # 0.9 looks nice, around 1

    plt.imshow(mask_segmentation_3bands)
    plt.show()
"""



main()




