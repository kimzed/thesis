# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 15:31:20 2021

@author: baron015


"""


import os
import geopandas as gpd
from shapely.geometry import MultiPolygon
import fiona
from shapely.geometry import shape
import datetime
import osmnx as ox

# google earth library
import ee

working_directory = "C:/Users/57834/Documents/thesis/"
os.chdir(working_directory)

import utils as functions
import pre_processing.utils_preprocessing as function_pre_processing

year = 2011
next_year = year + 1

folder_polygons_preprocess_year = f"pre_processing_data/labels_polygons_pre_processing/{year}/"
folder_polygons_preprocess = "pre_processing_data/labels_polygons_pre_processing/"

#code_ls8 = "LANDSAT/LC08/C01/T1_SR"
#code_ls7 = "LANDSAT/LE07/C01/T1_SR"
code_landsat = "LANDSAT/LE07/C01/T1_SR"


def main():

    print("""
          Prepare the grid for our ground truth and save it as shp
          """)


    shape_points_kenya = function_pre_processing.PointRegion('Kenya', 'epsg:4326')

    x_points_kenya = [point[0] for point in shape_points_kenya]
    y_points_kenya = [point[1] for point in shape_points_kenya]

    min_points_x, min_points_y = min(x_points_kenya), min(y_points_kenya)
    max_points_x, max_points_y = max(x_points_kenya), max(y_points_kenya)

    area_kenya_points = (min_points_x, min_points_y, max_points_x, max_points_y)
    size_tile_degree = 0.01
    tiles_grid_kenya = function_pre_processing.generates_list_tiles_from_square_points(area_kenya_points,
                                                                          size_tile_degree)

    grid_country = gpd.GeoDataFrame({'geometry':tiles_grid_kenya})

    epgs84_coordinate_system = ox.geocode_to_gdf("Kenya").crs
    grid_country.crs = epgs84_coordinate_system
    grid_country.to_file(f"{folder_polygons_preprocess_year}grid.shp")

    # =============================================================================
    # # calculation for the size_tile_degree # abs(36.948507 - 36.957975)
    # =============================================================================

    label_polygons = fiona.open(f'{folder_polygons_preprocess_year}Polygons.shp')
    multipolygon_labels = MultiPolygon([shape(polygon['geometry']) for polygon in label_polygons])


    tiles_touch_label_polygons_bool = grid_country.intersects(multipolygon_labels)
    tiles_including_label = grid_country[tiles_touch_label_polygons_bool]
    tiles_including_label.to_file(f"{folder_polygons_preprocess_year}grid_touch.shp")

    print("""
          Loading Landsat data and clipping around grid
          """)

    ee.Initialize()

    shape_points_kenya = function_pre_processing.PointRegion('Kenya', 'epsg:4326')
    points = [[point[0], point[1]] for point in shape_points_kenya]
    poly_kenya = ee.Geometry.Polygon(points)

    cloud_mask = function_pre_processing.mask_landsat8_cloud_shadow
    date_start = datetime.datetime(year, 1, 1)
    date_end = datetime.datetime(next_year, 1, 1)
    
    landsat_kenya = (ee.ImageCollection(code_landsat)
                 .filterDate(date_start,
                             date_end).filterBounds(poly_kenya)
                 .map(cloud_mask))
    landsat_kenya_median = landsat_kenya.median()

    landsat_images_has_label, geometry_tiles = function_pre_processing.clips_eeImage(landsat_kenya_median,
                            tiles_including_label)

    name_file_drive_GH = f"raster_GH_{year}_"
    function_pre_processing.upload_eeImages_to_drive(landsat_images_has_label, geometry_tiles,
                                       file_name_upload=name_file_drive_GH)

    file_polygon_nairobi_region = f'{folder_polygons_preprocess}Polygons_nairobi.shp'
    shp_nairobi = gpd.read_file(file_polygon_nairobi_region)
    
    # we add a buffer to avoid labels outside the image
    shp_nairobi = shp_nairobi.buffer(0.1)
    geometry_nairobi = shp_nairobi.geometry[0]
    points_nairobi = function_pre_processing.extract_points_poly(geometry_nairobi)
    ee_geometry_nairobi = ee.Geometry.Polygon(points_nairobi)
    landsat_image_nairobi_2019 = landsat_kenya_median.clip(ee_geometry_nairobi)
    
    name_file_drive_nairo = f"raster_nairo_{year}"
    function_pre_processing.UploadingEEData(landsat_image_nairobi_2019, name_file_drive_nairo, ee_geometry_nairobi)




### running the program
main()
