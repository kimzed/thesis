
import os
import geopandas as gpd
from shapely.geometry import MultiPolygon
import fiona
from shapely.geometry import shape
import rasterio
import rasterio.mask
import rasterio.mask
import numpy as np
from rasterio.plot import show
import matplotlib.pyplot as plt
import osmnx as ox
import datetime
from shapely.geometry import shape
import shutil

# google earth library
import ee

working_directory = "//WURNET.NL/Homes/baron015/My Documents/thesis/"
os.chdir(working_directory)

import python.pre_processing.utils_preprocessing as function_pre_processing
import python.utils as functions

import warnings
warnings.filterwarnings('ignore')


folder_data_preprocess = "pre_processing_data/inference_data_pre_processing"
foder_data_rasters = "inference_data/"

def main():
    
    
    
    print("""
          
          loading one raster and see what we get
          
          
          """)

    # for landsat 7 [0,1,2,3,4,6]
    # for landsat 8 [1,2,3,4,5,6]
    bands_selected =  [1,2,3,4,5,6]
    
    
    file_raster = f"{foder_data_rasters}raster_inference_2019_20_inference.tif"
          
    raster_test = rasterio.open(file_raster)
    
    raster_array = raster_test.read()
    
    raster_arrays = [raster_array]
    
    raster_arrays = [raster[bands_selected] for raster in raster_arrays]
    
    # the function also performs bilinear interpolation
    raster_arrays = function_pre_processing.resize_list_raster(raster_arrays)
    
    ## TODO save as numpy and check the size to see if it is reasonable or not
    
    print("""
          Loading the model
          """)
          
    folder_cnn_save = "runs/CnnModel/"

    path_best_model = f"{folder_cnn_save}2022_01_23_16_09.pt"
    #import python.model.CNN as CNN
    #model = CNN.CnnSemanticSegmentation()

    #model = load_weights_to_model(model, path_best_model)
    
    
    print("""
          Test the raster saving
          """)
          
    array_to_save_test = np.zeros([1, 6912, 6912])
    
    functions.save_array_as_raster(array_to_save_test, raster_test, file_raster)
    
        


main()