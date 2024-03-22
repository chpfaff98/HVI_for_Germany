# First module import
from time import time
start = time()
import os
os.system('clear')

# Printing user information in the command line
print()
print(" -------->         HVI computation for Germany         <--------")
print(" --------> by Christian Pfaff (chpfaff@uni-potsdam.de) <--------")
print()

# Importing python built-in modules
import sys
import shutil
from glob import glob

# Importing external modules
import dask.dataframe as dd
from geocube.api.core import make_geocube
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from osgeo import gdal
gdal.DontUseExceptions()
import pandas as pd
import rasterio as rio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from rasterio.merge import merge
from scipy.ndimage import distance_transform_edt
import seaborn as sns
from shapely.geometry import Polygon

# Filepath management
root = "/Users/hvi/" # Change here
in_dir = root + "input_data/"
p = root + "output_data/"

# Grid Creation output directories
out_dir_1_1 = p + "grids/germany/"
out_dir_1_2 = p + "grids/study_area/"

# Regional Processing output directories
out_dir_2_1 = p + "regional/census/filtered_csv/"
out_dir_2_2 = p + "regional/census/shp/"
out_dir_2_3 = p + "regional/raster/reprojected/"
out_dir_2_4 = p + "regional/raster/clipped/"
out_dir_2_5 = p + "regional/raster/resampled/tif/"
out_dir_2_6 = p + "regional/raster/resampled/shp/"
out_dir_2_7 = p + "regional/raster/resampled/csv/"
out_dir_2_8 = p + "regional/df/"
out_dir_2_9 = p + "regional/correlation/"
out_dir_2_10 = p + "regional/mean_std/"

# National Processing output directories
out_dir_3_1 = p + "national/census/"
out_dir_3_2 = p + "national/raster/preprocessed/"
out_dir_3_3 = p + "national/raster/reprojected/"
out_dir_3_4 = p + "national/raster/resampled/"
out_dir_3_5 = p + "national/raster/masked/"
out_dir_3_6 = p + "national/mean_std/"

# HVI output directories
out_dir_4_1 = p + "hvi/regional/"
out_dir_4_2 = p + "hvi/national/"
out_dir_4_3 = p + "hvi/difference/"

# Validation output directories
out_dir_5_1 = p + "validation/preprocessed/"
out_dir_5_2 = p + "validation/regional/shp/"
out_dir_5_3 = p + "validation/regional/csv/"
out_dir_5_4 = p + "validation/national/shp/"
out_dir_5_5 = p + "validation/national/csv/"

# Putting all directory names in a list
out_paths = [out_dir_1_1, out_dir_1_2, out_dir_2_1, out_dir_2_2, out_dir_2_3,
             out_dir_2_4, out_dir_2_5, out_dir_2_6, out_dir_2_7, out_dir_2_8,
             out_dir_2_9, out_dir_2_10, out_dir_3_1, out_dir_3_2, out_dir_3_3,
             out_dir_3_4, out_dir_3_5, out_dir_3_6, out_dir_4_1, out_dir_4_2,
             out_dir_4_3, out_dir_5_1, out_dir_5_2, out_dir_5_3, out_dir_5_4, 
             out_dir_5_5]

# Creating the output directories
for i in out_paths:
    if not os.path.exists(i):
        os.makedirs(i)

# Finish initiation process
print("-- initiation process finished in {:.2f} s --".format(time()-start))
print()