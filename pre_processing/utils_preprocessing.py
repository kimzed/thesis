
import os
import rasterio
import rasterio.mask
import numpy as np
from shapely.geometry import box
import rasterio.mask
import osmnx as ox
from shapely.geometry import mapping
import geopandas


# google earth library
import ee
from ee import batch

working_directory = "//WURNET.NL/Homes/baron015/My Documents/thesis"
os.chdir(working_directory)

folder_polygons_preprocess = "pre_processing_data/labels_polygons_pre_processing/"
folder_rasters_preprocess = "pre_processing_data/rasters/"


import python.utils as functions



#### Functions


def converting_non_zeros_to_ones(arrays: list):
    arrays_result = []

    for array in arrays:
        array[array != 0] = 1
        arrays_result.append(array)

    return arrays_result


def masking_rasters_on_multipolygon(multi_polygon, rasters: list):
    cropped_rasters_result = []

    for raster in rasters:
        cropped_raster, _ = rasterio.mask.mask(raster, multi_polygon, crop=True)
        cropped_rasters_result.append(cropped_raster)

    return cropped_rasters_result


def resize_list_raster(rast_np):
    ## linear interpolation to get the same height and width
    # defining the new size
    sizes = [rast.shape[1] for rast in rast_np]
    new_size = int(np.floor(np.mean(sizes)))

    for i in range(len(rast_np)):

        # creating a new raster to store landsat interpolated data
        nb_bands = rast_np[0].shape[0]
        new_rast = np.zeros((nb_bands, new_size, new_size))

        ## performing interpolation to remove nan values
        if str(rast_np[i].mean()) == "nan":

            # looping over the bands
            for b in range(rast_np[i].shape[0]):
                # extract the band
                rast = rast_np[i][b, :, :]

                # get the mask of nan values
                mask_nan = np.isnan(rast)
                cl_rast = functions.interpolate_missing_pixels(rast, mask_nan)

                # updating the raster
                rast_np[i][b, :, :] = cl_rast

        # interpolation along the different bands
        for b in range(new_rast.shape[0]):
            new_rast[b, :, :] = functions.regrid(rast_np[i][b, :, :], new_size, new_size)

            # apply normalization
            # new_rast[b,:,:] -= new_rast[b,:,:].mean()

        # updating the raster
        rast_np[i] = new_rast

    return rast_np


def regrid_label(raster, size):
    """
    Function to regrid and flatten the labels (discrete data)
    """

    # reshaping
    rast_resh = functions.regrid(raster.reshape(raster.shape), size, size, "nearest")

    # converting again back to integers (reshaping generates floats)
    rast_resh = np.rint(rast_resh)

    rast_resh = rast_resh.flatten()

    return rast_resh


def masking_polygons_on_raster(polygons: list, raster, bands: list):
    rasters_output = []

    for polygon in list(polygons.geometry):
        cropped_raster, _ = rasterio.mask.mask(raster, [polygon], crop=True, all_touched=True)
        rasters_output.append(cropped_raster[bands, :, :])

    return rasters_output


def UploadingEEData(ee_image, name_file, poly):
    '''
    arg:
    fun:
    '''

    # Creating the exporting object
    out = batch.Export.image.toDrive(image=ee_image,
                                     scale=30,
                                     folder='earthengine',
                                     fileNamePrefix=name_file,
                                     maxPixels=1e9,
                                     region=poly)

    # Process the image
    process = batch.Task.start(out)


def CentroidPoints(df):
    '''
    arg: df as a multipolygon geodf
    fun: returns a list of nested points of the centroid of each polygons
    '''
    # Getting the points as a multipoints df
    multi_points = df.centroid

    # Getting the points as a list
    list_points = multi_points.tolist()

    # Getting a list of coordinates
    coordinates = [list(point.coords) for point in list_points]

    # Denest the list
    coordinates_denested = [point[0] for point in coordinates]

    return coordinates_denested


def mask_landsat8_cloud_shadow(image_collection):
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image_collection.select('pixel_qa')

    maximum_percentage = 10
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).lt(maximum_percentage)
    mask2 = qa.bitwiseAnd(cloudsBitMask).lt(maximum_percentage)

    # applying both masks
    image = image_collection.updateMask(mask)
    image = image.updateMask(mask2)

    return image.updateMask(mask)


def GetOSMdata(placename, pr_code):
    '''
    arg: two strings, one for the place the other for the projection
    fun: extracts osm data and returns it as a dataframe
    '''
    # Making the variable
    vector = ox.geocode_to_gdf(placename)

    # Changing the vector
    vector_crs = vector.to_crs({'init': pr_code})

    return vector_crs


def PointRegion(region, epsg):
    '''
    arg: region as a string, epsg as a string
    fun: returns a list of points
    '''
    # Getting the geographical data
    data = GetOSMdata(region, epsg)

    # Extracting points
    geo_data = mapping(data)
    features = geo_data['features'][0]
    poly = features['geometry']
    nested_points = list(poly['coordinates'])

    # Denesting the points
    points = [point for point in nested_points[0]]

    return points


def upload_eeImages_to_drive(eeimages: list, eegeometris: list, file_name_upload):
    
    i_image = 0

    for image, geometry in zip(eeimages, eegeometris):
        name_file_upload = file_name_upload + str(i_image)
        UploadingEEData(image, name_file_upload, geometry)
        
        i_image += 1


def clips_eeImage(eeImage: ee.image.Image, multi_polygon: geopandas.geodataframe.GeoDataFrame):
    eeimages_clipped = []
    eegeometries = []

    for poly in multi_polygon.geometry:
        points = extract_points_poly(poly)
        eegeometry_poly = ee.Geometry.Polygon(points)
        clipped_eeImage = eeImage.clip(eegeometry_poly)
        eeimages_clipped.append(clipped_eeImage)
        eegeometries.append(eegeometry_poly)

    return eeimages_clipped, eegeometries


def generates_list_tiles_from_square_polygon(polygon_points: tuple, size_tile):
    min_x, min_y, max_x, max_y = polygon_points
    moving_x = min_x
    moving_y = min_y
    tiles = []

    while moving_x <= max_x:
        moving_y = min_y

        while moving_y <= max_y:
            upper_bound_tile = moving_x + size_tile
            right_bound_tile = moving_y + size_tile

            bbox = box(moving_x, moving_y,
                       upper_bound_tile, right_bound_tile)
            tiles.append(bbox)
            moving_y += size_tile

        moving_x += size_tile

    return tiles


def extract_points_poly(poly):
    # get the points
    points = poly.exterior.coords.xy

    # getting longitude and latitude
    long = points[0]
    lati = points[1]

    # putting into a single list
    points = [[lon, lat] for lon, lat in zip(long, lati)]

    return points