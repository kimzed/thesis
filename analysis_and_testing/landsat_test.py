# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:29:33 2021

@author: baron015
"""

# Import modules
import os
import sys
import pandas as pd
import numpy as np
import ee
from ee import batch
import osmnx as ox
from shapely.geometry import mapping
import datetime
import json
import geojson
import tempfile
import geopandas as gpd
from rasterio.plot import show



## changing working directory
os.chdir("//WURNET.NL/Homes/baron015/My Documents/thesis")

def UploadingEEData(ee_image, name_file, poly):
    '''
    arg:
    fun:
    '''
    
    # Creating the exporting object
    out = batch.Export.image.toDrive(image=ee_image,
                                     scale=10,
                                     folder= 'earthengine',
                                     fileNamePrefix=name_file,
                                     maxPixels= 1e9,
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
    

def GetOSMdata(placename, pr_code):
    '''
    arg: two strings, one for the place the other for the projection
    fun: extracts osm data and returns it as a dataframe
    '''
    # Making the variable
    vector = ox.geocode_to_gdf(placename)

    # Changing the vector
    vector_crs = vector.to_crs({'init' : pr_code})

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

def maskL8sr(image):
    """
    function that takes a landsat EE image and get only the pixels with 0 clouds
    or 0 shadow
    """
    
    # Bits 3 and 5 are cloud shadow and cloud, respectively.
    cloudShadowBitMask = (1 << 3)
    cloudsBitMask = (1 << 5)
    # Get the pixel QA band.
    qa = image.select('pixel_qa')
    # Both flags should be set to zero, indicating clear conditions.
    mask = qa.bitwiseAnd(cloudShadowBitMask).eq(0)
    mask2 = qa.bitwiseAnd(cloudsBitMask).eq(0)
    
    # applying both masks
    image = image.updateMask(mask)
    image = image.updateMask(mask2)
    
    return image.updateMask(mask)


### get the grid for the country


## connecting to the google earth engine account
ee.Initialize()


### get the polygon of the whole country
# Extracting the points of our regions as a list of tuples
points_kenya = PointRegion('Kenya', 'epsg:4326')

### getting the geojson
points = [[point[0], point[1]] for point in points_kenya]

poly_kenya = ee.Geometry.Polygon(points)



#### we test with a smaller polygon
shpGDF = gpd.read_file('data_labels/Polygons_1.shp')

points_gdf = CentroidPoints(shpGDF)

poly_gh = ee.Geometry.Polygon(points_gdf)

# Creating the ee image object
landsat = (ee.ImageCollection('LANDSAT/LC08/C01/T1_SR')
             .filterDate(datetime.datetime(2019, 1, 1),
                         datetime.datetime(2020, 1, 1)).filterBounds(poly_gh)
             .map(maskL8sr))

# converting to an image
LS_img = landsat.median()
            
# Uploading the 2000 data
UploadingEEData(LS_img, 'test_raster1', poly_gh)




# =============================================================================
# geo_json = [ {"type": "Feature",
#               "geometry": {
#                   "type": "Point",
#                   "coordinates": [coord[0], coord[1]] }}
#               for coord in points_kenya ] 
# 
# geo_str = json.dumps(geo_json)
# 
# =============================================================================
