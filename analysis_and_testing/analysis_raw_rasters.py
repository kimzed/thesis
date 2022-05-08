import os
import geopandas as gpd
from shapely.geometry import MultiPolygon
import fiona
from shapely.geometry import shape
import rasterio
import rasterio.mask
import rasterio.mask
import numpy as np

import warnings
warnings.filterwarnings('ignore')


working_directory = "C:/Users/57834/Documents/thesis"
os.chdir(working_directory)

# possible years: 2019, 2014, 2011
years = [2011, 2014, 2019]



import python.utils as functions
import python.pre_processing.utils_preprocessing as function_pre_processing

folder_results = "maps_and_results/"
file_results = f"{folder_results}results_data_analysis.txt"




def main():
    
    for year in years:
                
        folder_rasters_preprocess = f"pre_processing_data/rasters/{year}/"




        print("""
                  Loading the rasters and creating the label data
                  """)
    
        file_nairobi_raster = f"{folder_rasters_preprocess}raster_nairo_{year}.tif"
    
        raster_nairobi = rasterio.open(file_nairobi_raster)
        array_nairobi = raster_nairobi.read()
        
        # keeping only one band
        array_nairobi = array_nairobi[0,:,:]
        
        folder_rasters_greenhouses = f"{folder_rasters_preprocess}landsat_rasters_greenhouses"
        raster_files = functions.get_files(folder_rasters_greenhouses)
    
        arrays_greenhouses = [rasterio.open(file).read() for file in raster_files]
        
        print(""" 
              analysis number of nans nairobi
              """)
              
        is_nan_nairobi = np.isnan(array_nairobi)
        is_nan_nairobi = is_nan_nairobi.flatten()
        number_nan_nairobi = np.count_nonzero(is_nan_nairobi == True)
        
        result = f"for nairobi on year {year}, there are {number_nan_nairobi} nan values, for {len(is_nan_nairobi)} in total \n\n\n"
        
        functions.write_to_txt_file(file_results, result)
        
        print(""" 
              analysis number of nans greenhouse samples
              """)
        list_number_nans = []
        
        for array_greenhouse in arrays_greenhouses:
            is_nan_raster = np.isnan(array_greenhouse)
            is_nan_raster = is_nan_raster.flatten()
            number_nan_raster = np.count_nonzero(is_nan_raster == True)
            
            list_number_nans.append(number_nan_raster)
        
        array_number_nans = np.array(list_number_nans)
        mean_number_nans = array_number_nans.mean()
        std_number_nans = array_number_nans.std()
        
        result = f"for greenhouse samples on year {year}, mean nans is {mean_number_nans} and std is {std_number_nans} \n\n\n"
        
        functions.write_to_txt_file(file_results, result)

try:
    main()
except Exception as exception:

    print(exception)
    raise exception