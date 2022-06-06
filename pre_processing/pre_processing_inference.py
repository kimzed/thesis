import os
import geopandas as gpd
import fiona
import osmnx as ox
import datetime
from shapely.geometry import shape

# google earth library
import ee


import utils_preprocessing as function_pre_processing

import warnings

warnings.filterwarnings('ignore')

folder_data_preprocess = "pre_processing_data/inference_data_pre_processing"
foder_data_rasters = "inference_data/"

# year for the landsat raster
years = [2020]

code_ls8 = "LANDSAT/LC08/C01/T1_SR"
code_ls7 = "LANDSAT/LE07/C01/T1_SR"
code_landsat = code_ls8

if code_landsat == code_ls8:
    bands_to_select = ["B2", "B3", "B4", "B5", "B6", "B7"]
else:
    bands_to_select = ["B1", "B2", "B3", "B4", "B5", "B7"]


def main():
    print("""
          Making a grid on the whole country
          """)

    shape_points_kenya = function_pre_processing.PointRegion('Kenya', 'epsg:4326')

    x_points_kenya = [point[0] for point in shape_points_kenya]
    y_points_kenya = [point[1] for point in shape_points_kenya]

    min_points_x, min_points_y = min(x_points_kenya), min(y_points_kenya)
    max_points_x, max_points_y = max(x_points_kenya), max(y_points_kenya)

    size_tile_degree = 0.8
    area_kenya_points = (min_points_x, min_points_y, max_points_x, max_points_y)

    tiles_grid_kenya = function_pre_processing.generates_list_tiles_from_square_points(area_kenya_points,
                                                                                       size_tile_degree)

    gridCountry = gpd.GeoDataFrame({'geometry': tiles_grid_kenya})
    epgs84_coordinate_system = ox.geocode_to_gdf("Kenya").crs
    gridCountry.crs = epgs84_coordinate_system
    gridCountry.to_file("pre_processing_data/inference_data_pre_processing/grid_country_inference.shp")

    # select only the ones that touch kenya
    file_kenya_polygon = "pre_processing_data/polygon_kenya/gadm36_KEN_0.shp"
    polygonKenya = fiona.open(file_kenya_polygon)
    polygonKenyaGeometry = shape(polygonKenya[0]['geometry'])
    tiles_touches_kenya = gridCountry.intersects(polygonKenyaGeometry)

    gridCountry = gridCountry[tiles_touches_kenya]

    number_tiles = len(gridCountry)
    print(f"the grid has {number_tiles} tiles")

    print("""
          Loading Landsat data and clipping around grid
          """)

    for year in years:
        next_year = year + 1

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

        landsat_kenya = landsat_kenya.select(bands_to_select)
        landsat_kenya_median = landsat_kenya.median()

        landsat_images, geometry_tiles = function_pre_processing.clips_eeImage(landsat_kenya_median,
                                                                               gridCountry)

        name_file_drive_GH = f"earth_engine/{year}/raster_inference_{year}"
        function_pre_processing.upload_eeImages_to_drive(landsat_images, geometry_tiles,
                                                         file_name_upload=name_file_drive_GH)


main()
