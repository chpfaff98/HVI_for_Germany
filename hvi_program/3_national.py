# importing modules
from helper_module import *
from helper_module.functions import *
start = time()

# Input parameters
mask_fp = p + "grids/germany/germany_100m_bevoelkerungs_grid.tif"
demographic_fp = in_dir + "csv_Demographie_100m_Gitter/Bevoelkerung100M.csv"
household_fp = in_dir + "csv_Haushalte_100m_Gitter/Haushalte100M.csv"
lst_fp = in_dir + "raster_data/LST_BRD_median_gapFilled_2018_2023_clip.tif"
gv_fp = in_dir + "raster_data/GV_RF_Regression_Deutschland_multitemp_2022.tif"
sea_22_fp = in_dir + "raster_data/Versiegelung_RF_Regression_Deutschland_multitemp_2022_clip.tif"
sea_23_fp = in_dir + "raster_data/Versiegelung_RF_Regression_Deutschland_multitemp_2023.tif"

# ------------------- Census Filtering for entire Germany -------------------

if False: # National filtering of the census data
    census_data = [demographic_fp, demographic_fp, demographic_fp, household_fp]
    merkmal = ['ALTER_KURZ', 'STAATSANGE_KURZ', 'ALTER_10JG', 'HHGROESS_KLASS']
    code = [5, 2, 1, 1]
    for i in range(len(census_data)):
        census_filter_ger(census_data[i], merkmal[i], code[i], out_dir_3_1)

# ----------- Preprocessing the Sealing Germany 2022/2023 raster ------------

if False: # Clipping the 2023 sea raster to the extent of the 2022 sea raster
    tif_extent_cropper(sea_23_fp, sea_22_fp, out_dir_3_2)

if False: # Filling gaps of the sea 2023 raster with the sea 2022 raster
    sea_23_cropped_fp = glob(out_dir_3_2 + "*_cropped.tif")[0]
    tif_filler(sea_23_cropped_fp, 128, sea_22_fp, out_dir_3_2)

if False: # Filling remaining gaps of Sealing 2023 with Euclidean Distance
    sea_23_cropped_filled_fp = glob(out_dir_3_2 + "*_cropped_filled.tif")[0]
    tif_edt_filler(sea_23_cropped_filled_fp, out_dir_3_2, delete=True)

# --------------------- Raster Resampling for entire Germany ------------------

if False: # Reprojecting german-wide GV and filled SEA raster
    filepaths = [gv_fp, lst_fp, glob(out_dir_3_2 + "*_nb_filled.tif")[0]]
    for i in filepaths:
        # The following function was first introduced in 2_regional.py
        tif_reprojector(i, out_dir_3_3)

if False: # Resampling all reprojected raster to Germany Population Grid
    filepaths = [glob(out_dir_3_3 + "LST*.tif")[0],
                 glob(out_dir_3_3 + "GV_RF_*.tif")[0],
                 glob(out_dir_3_3 + "Versiegelung*.tif")[0]]
    nan_value = [0, 65536, None]
    nodata_flag = [True, True, True]
    for i in range(len(filepaths)):
        tif_resampler_national(filepaths[i], mask_fp, nan_value[i], Resampling.average, 
                      out_dir_3_4, nodata_flag[i])

if False: # Masking all the raster to the Germany Population Grid
    raster_res_fp = glob(out_dir_3_4 + "*.tif")
    for i in raster_res_fp:
        tif_masker(i, mask_fp, out_dir_3_5)

# -------------------- Mean and Std for the entire Germany --------------------

if False:
    mean_std_ger_calculator(out_dir_3_6)

# -------------------------------------------------------------------------

# Finishing the script
print("Script finished in: {:.2f} s".format(time()-start))
print()