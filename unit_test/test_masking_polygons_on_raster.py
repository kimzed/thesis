from pathlib import Path
import numpy as np
import os

from pre_processing.utils_preprocessing import masking_polygons_on_raster
from shapely.geometry import MultiPolygon
from shapely.geometry import shape
import fiona
import rasterio
import pre_processing.utils_preprocessing as function_pre_processing

import geopandas as gpd

year = 2019
folder_polygons_preprocess_year = f"pre_processing_data/labels_polygons_pre_processing/{year}/"
folder_polygons_preprocess = f"pre_processing_data/labels_polygons_pre_processing/"
folder_rasters_preprocess = f"pre_processing_data/rasters/{year}/"


def get_files(dir_files):
    """
    Get all the files from the dir in a list with the complete file path
    Get all the files from the dir in a list with the complete file path
    """

    # empty list to store the files
    list_files = []

    # getting the files in the list
    for path in os.listdir(dir_files):
        full_path = os.path.join(dir_files, path)
        if os.path.isfile(full_path):
            list_files.append(full_path)

    return list_files


def masking_polygons_on_raster(polygons, raster: rasterio.io.DatasetReader, bands: list) -> np.array:
    rasters_output = []

    for polygon in polygons:
        cropped_raster, _ = rasterio.mask.mask(raster, [polygon], crop=True, all_touched=True)
        rasters_output.append(cropped_raster[bands, :, :])

    return rasters_output


def test_masking_polygons_on_raster():
    file_grid_country = f'{folder_polygons_preprocess_year}grid.shp'
    grid_country = gpd.read_file(file_grid_country)

    file_nairobi_raster = f"{folder_rasters_preprocess}raster_nairo_{year}.tif"

    raster_nairobi = rasterio.open(file_nairobi_raster)
    file_polygons_nairobi = f'{folder_polygons_preprocess}Polygons_nairobi.shp'
    polygon_nairobi = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(file_polygons_nairobi)])

    tiles_touch_nairobi = grid_country.intersects(polygon_nairobi)
    grid_nairobi_dataset = grid_country[tiles_touch_nairobi]

    gt_rasters_greenhouse = function_pre_processing.masking_polygons_on_raster(grid_nairobi_dataset, raster_nairobi,
                                                                               bands=[0])


def test_masking_polygons_on_raster_using_greenhouse_polygons():
    file_polygons_labels = f'{folder_polygons_preprocess_year}Polygons.shp'
    polygons_labels = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(file_polygons_labels)])

    folder_rasters_greenhouses = f"{folder_rasters_preprocess}landsat_rasters_greenhouses"
    raster_files = get_files(folder_rasters_greenhouses)

    file_nairobi_raster = f"{folder_rasters_preprocess}raster_nairo_{year}.tif"

    raster_nairobi = rasterio.open(file_nairobi_raster)

    polygons = list(polygons_labels)
    gt_rasters_greenhouse = masking_polygons_on_raster(polygons_labels, raster_nairobi, bands=[0])


def test_masking_rasters_on_multipolygon():
    file_polygons_labels = f'{folder_polygons_preprocess_year}Polygons.shp'
    polygons_labels = MultiPolygon([shape(pol['geometry']) for pol in fiona.open(file_polygons_labels)])

    polygons_labels = function_pre_processing.remove_third_dimension(polygons_labels)

    folder_rasters_greenhouses = f"{folder_rasters_preprocess}landsat_rasters_greenhouses"
    raster_files = get_files(folder_rasters_greenhouses)

    rasters_greenhouses = [rasterio.open(file) for file in raster_files]

    polygons = list(polygons_labels)

    gt_rasters_greenhouse = function_pre_processing.masking_rasters_on_multipolygon(polygons_labels,
                                                                                    rasters_greenhouses)


#test_masking_polygons_on_raster()
#test_masking_polygons_on_raster_using_greenhouse_polygons()
test_masking_rasters_on_multipolygon()
