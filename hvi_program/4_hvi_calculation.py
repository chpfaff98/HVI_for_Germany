# importing modules
from helper_module import *
from helper_module.functions import *
start = time()

# Input parameters
df_fp = out_dir_2_8 + "df_uncorrelated.csv" 
sa_stats_fp = out_dir_2_10 + "mean_std_sa.csv"
ger_stats_fp = out_dir_3_6 + "mean_std_ger.csv"

# --------------------- HVI with regional mean and std ---------------------

if False:
    hvi_calculator(df_fp, sa_stats_fp, out_dir_4_1)

# --------------------- HVI with national mean and std ---------------------

if False:
    hvi_calculator(df_fp, ger_stats_fp, out_dir_4_2)

# --------------------- Calculating the HVI Difference ---------------------

if False: # National HVI - Regional HVI
    shp_hvi_diff(glob(out_dir_4_1+"*.shp")[0], glob(out_dir_4_2+"*.shp")[0],
                 out_dir_4_3)

# -------------------------------------------------------------------------

# Finishing the script
print("Script finished in: {:.2f} s".format(time()-start))
print()