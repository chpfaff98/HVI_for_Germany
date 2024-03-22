# Importing modules
from helper_module import *
from helper_module.functions import *
start = time()

# Input parameters
shp_study_area_fp = in_dir + "Geobroker/data/GRENZE_348591-5799606_gemeinden.shp"
bevoelkerung_fp = in_dir + "csv_Bevoelkerung_100m_Gitter/Zensus_Bevoelkerung_100m-Gitter.csv"

# ---------------------------- Germany-wide Grid -----------------------------

if False: # Creation of Germany Population Grid
    germany_grid_creator(bevoelkerung_fp, out_dir_1_1)

if False: # Rasterisation of the Germany Puplation Grid
    shp_fp = out_dir_1_1 + "germany_100m_bevoelkerungs_grid.shp"
    raster_mask_creator(shp_fp, out_dir_1_1)

# ------------------------- Potsdam Grid Creation -------------------------

if False: # Reprojecting and Filtering the Administrative Borders
    shp_study_area_preprocessor(shp_study_area_fp, out_dir_1_2)

if False: # Creating of th Buffered Border of Potsdam (150m)
    shp_study_area_fp = out_dir_1_2 + "study_area.shp"
    study_area_buffer_creator(shp_study_area_fp, 150, out_dir_1_2)

if False: # Spatial Intersection of the Germany Population grid and Border of Potsdam
    grid_ger_fp = out_dir_1_1 + "germany_100m_bevoelkerungs_grid.shp"
    study_area_fp = out_dir_1_2 + "study_area.shp"
    study_area_masking(grid_ger_fp, study_area_fp, out_dir_1_2)

if False: # Rasteristation of the Potsdam Population Grid
    shp_fp = out_dir_1_2 + "study_area_grid_100m.shp"
    raster_mask_creator(shp_fp, out_dir_1_2)

# -------------------------------------------------------------------------

# Finishing the script
print("Script finished in: {:.2f} s".format(time()-start))
print()