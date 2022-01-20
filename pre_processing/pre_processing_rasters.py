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


working_directory = "//WURNET.NL/Homes/baron015/My Documents/thesis"
os.chdir(working_directory)

# possible years: 2019, 2014, 2011
year = 2011
# for landsat 7 [0,1,2,3,4,6]
# for landsat 8 [1,2,3,4,5,6]
bands_selected =  [0,1,2,3,4,6]


folder_polygons_preprocess_year = f"pre_processing_data/labels_polygons_pre_processing/{year}/"
folder_polygons_preprocess = f"pre_processing_data/labels_polygons_pre_processing/"
folder_rasters_preprocess = f"pre_processing_data/rasters/{year}/"



import python.utils as functions
import python.pre_processing.utils_preprocessing as function_pre_processing

def main():
    print("""
              Loading the rasters and creating the label data
              """)

    file_nairobi_raster = f"{folder_rasters_preprocess}raster_nairo_{year}.tif"
    file_grid_country = f'{folder_polygons_preprocess_year}grid.shp'
    file_polygons_labels = f'{folder_polygons_preprocess_year}Polygons.shp'
    file_polygons_nairobi = f'{folder_polygons_preprocess}Polygons_nairobi.shp'

    raster_nairobi = rasterio.open(file_nairobi_raster)
    grid_country = gpd.read_file(file_grid_country)
    polygon_nairobi = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(file_polygons_nairobi)])
    polygons_labels = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(file_polygons_labels)])

    tiles_touch_nairobi = grid_country.intersects(polygon_nairobi)
    tiles_touch_labels = grid_country.intersects(polygons_labels)

    # removing the cells that touch the labels
    tiles_touch_nairobi[tiles_touch_labels] = False
    grid_nairobi_dataset = grid_country[tiles_touch_nairobi]

    
    rasters_nairobi = function_pre_processing.masking_polygons_on_raster(grid_nairobi_dataset,
                                                 raster_nairobi,
                                                 bands_selected)

    

    # subset_nair = False

    # if subset_nair:
    # subsetting to reduce the number of samples
    # gt_list, rast_np = zip(*random.sample(list(zip(gt_list, rast_np)), len(rast_np)//10))
    # gt_list = list(gt_list)
    # rast_np = list(rast_np)

    rasters_nairobi = function_pre_processing.resize_list_raster(rasters_nairobi)
    gt_raster_nairobi = [np.zeros(raster.shape) for raster in rasters_nairobi]
    gt_raster_nairobi = [raster[0, :, :] for raster in gt_raster_nairobi]
    gt_raster_nairobi = [raster.flatten() for raster in gt_raster_nairobi]
    new_size = rasters_nairobi[0].shape[1]

    rasters_nairobi = [raster - raster.mean() for raster in rasters_nairobi]

    ### working with the greenhouse rasters
    folder_rasters_greenhouses = f"{folder_rasters_preprocess}landsat_rasters_greenhouses"
    raster_files = functions.get_files(folder_rasters_greenhouses)

    rasters_greenhouses = [rasterio.open(file) for file in raster_files]

    gt_rasters_greenhouse = function_pre_processing.masking_rasters_on_multipolygon(polygons_labels,
                                                            rasters_greenhouses)

    # keeping only one band
    gt_rasters_greenhouse = [raster[0, :, :] for raster in gt_rasters_greenhouse]
    gt_rasters_greenhouse = function_pre_processing.converting_non_zeros_to_ones(gt_rasters_greenhouse)

    arrays_greenhouses = [raster.read() for raster in rasters_greenhouses]
    # we only want seven bands
    arrays_greenhouses = [raster[bands_selected] for raster in arrays_greenhouses]

    arrays_greenhouses = function_pre_processing.resize_list_raster(arrays_greenhouses)
    arrays_greenhouses = [raster - raster.mean() for raster in arrays_greenhouses]
    new_size = arrays_greenhouses[0].shape[1]
    gt_rasters_greenhouse = [function_pre_processing.regrid_label(raster, new_size) for raster in gt_rasters_greenhouse]
        

    
    
    print("""
              Saving the data as numpy objects
              """)

    folder_greenhouse_dataset = f"data/{year}/greenhouse_dataset/"
    path_greenhouse_rasters = f"{folder_greenhouse_dataset}landsat_rasters/"
    path_greenhouse_gt_rasters = f"{folder_greenhouse_dataset}ground_truth_rasters/"
    
    functions.create_directory(f"data/{year}")
    functions.create_directory(folder_greenhouse_dataset)
    functions.create_directory(path_greenhouse_rasters)
    functions.create_directory(path_greenhouse_gt_rasters)

    functions.save_numpy_data(arrays_greenhouses, path=path_greenhouse_rasters, file_name="landsat_raster")
    functions.save_numpy_data(gt_rasters_greenhouse, path=path_greenhouse_gt_rasters, file_name="gt_raster")

    folder_nairobi_dataset = f"data/{year}/nairobi_negatives_dataset/"
    path_nairobi_rasters = f"{folder_nairobi_dataset}landsat_rasters/"
    path_nairobi_gt_rasters = f"{folder_nairobi_dataset}ground_truth_rasters/"
    
    functions.create_directory(folder_nairobi_dataset)
    functions.create_directory(path_nairobi_rasters)
    functions.create_directory(path_nairobi_gt_rasters)

    functions.save_numpy_data(rasters_nairobi, path=path_nairobi_rasters, file_name="landsat_raster")
    functions.save_numpy_data(gt_raster_nairobi, path=path_nairobi_gt_rasters, file_name="gt_raster")

main()