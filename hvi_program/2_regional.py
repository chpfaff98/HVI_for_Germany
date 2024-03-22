# importing modules
from helper_module import *
from helper_module.functions import *
start = time()

# Input parameters
grid_fp = out_dir_1_2 + 'study_area_grid_100m.shp'
mask_sa_fp = out_dir_1_2 + 'study_area_grid_100m.tif'
sa_buffer_shp_fp = out_dir_1_2 + 'study_area_150m_buffer.shp'
demographic_fp = in_dir + 'csv_Demographie_100m_Gitter/Bevoelkerung100M.csv'
household_fp = in_dir + 'csv_Haushalte_100m_Gitter/Haushalte100M.csv'
raster_fp = glob(in_dir + "raster_data/*otsdam*.tif")
raster_fp.append(in_dir + "raster_data/LST_BRD_median_gapFilled_2018_2023_clip.tif")

# ---------------------- Census Filtering for Potsdam ------------------------

if False: # Regional Filtering of the Demographic and Household Census data
    census_data = [demographic_fp, demographic_fp, demographic_fp, household_fp]
    merkmal = ['ALTER_KURZ', 'STAATSANGE_KURZ', 'ALTER_10JG', 'HHGROESS_KLASS']
    code = [5, 2, 1, 1]
    for i in range(len(census_data)):
        census_filter_sa(grid_fp, census_data[i], merkmal[i], code[i], out_dir_2_1)

if False: # Spatial Join of all Census variables based on Grid IDs
    csv_fp = glob(out_dir_2_1 + "*.csv")
    census_csv_merger(csv_fp, out_dir_2_1)

if False: # Creation of Shapefiles for each Census variable
    csv_fp = glob(out_dir_2_1 + "*.csv")
    csv_fp.remove(out_dir_2_1 + "census_merged.csv")
    for i in range(len(csv_fp)):
        shp_creator(csv_fp[i], grid_fp, "left", out_dir_2_2)

# ------------------------ Raster Processing for Potsdam ----------------------

if False: # Reprojecting the Raster to EPSG:3035
    for i in raster_fp:
        tif_reprojector(i, out_dir_2_3)

if False: # Clipping to Buffered Border of Potsdam
    raster_reproj_fp = glob(out_dir_2_3 + "*.tif")
    for i in raster_reproj_fp:
        tif_clipper(i, sa_buffer_shp_fp, out_dir_2_4)

if False: # Average Resampling to Potsdam Population Grid
    raster_sa = glob(out_dir_2_4 + "*.tif")
    for i in raster_sa:
        tif_resampler_regional(i, mask_sa_fp, Resampling.average, out_dir_2_5)

if False: # Spatial Joining to the Potsdam Population Grid
    raster_res_fp = glob(out_dir_2_5 + "*.tif")
    for i in raster_res_fp:
        tif_to_shp_joiner(i, grid_fp, out_dir_2_6)

if False: # Saving as csv files
    shp_resampled_fp = glob(out_dir_2_6 + "*.shp")
    for i in range(len(shp_resampled_fp)):
        shp_res_csv_saver(shp_resampled_fp[i], out_dir_2_7)

# ----------- Merging & correcting the Raster and Census csv files -------------

if False: # Spatial Merging of the CSV files
    census_merged_csv_fp= glob(out_dir_2_1 + "census_merged.csv")[0]
    csv_resampled_fp = glob(out_dir_2_7 + "*.csv")
    df_merger(csv_resampled_fp, census_merged_csv_fp, out_dir_2_8)

# ------------------- Spearman Rank Correlation for the Study Area -------------

if False: # Performing the Spearman Rank Correlation Analysis
    csv_fp = out_dir_2_8 + "df.csv"
    spearman_correlation_analysis_sa(csv_fp, out_dir_2_9)

# Due to the Spearman Rank Correlation Analysis, the following variables were
# dropped from the analysis: Vegetation Height and Canopy Cover

if False: # Removing very strong correlated variables
    csv_fp = out_dir_2_8 + "df.csv"
    dataframe_column_dropper(csv_fp, ["VH", "CP"], out_dir_2_8)

# -------------------- Mean and Std for regional Data Variables ----------------

if False: # Calculating the Regional Mean and Std for Potsdam
    df_fp = out_dir_2_8 + "df_uncorrelated.csv"
    mean_std_sa_calculator(df_fp, out_dir_2_10)

# ---------------------------------------------------------------------------

# Finishing the script
print("Script finished in: {:.2f} s".format(time()-start))
print()