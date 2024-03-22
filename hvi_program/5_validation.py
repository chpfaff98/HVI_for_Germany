# importing modules
from helper_module import *
from helper_module.functions import *
start = time()

# Input parameters and containers
val_raw_data = in_dir + "validation_data/siedlungsraum_block_rev00.shp"
hvi_data = [out_dir_4_1 + "FINAL_HVI_SA.shp", out_dir_4_2 + "FINAL_HVI_GER.shp"]
out_shp_fp = [out_dir_5_2, out_dir_5_4]
out_csv_fp = [out_dir_5_3, out_dir_5_5]

# ---------------------------- Preprocessing Validation Data ----------------------------

if False: # Reprojecting and filtering the validation data
    shp_validation_preprocessor(val_raw_data, out_dir_5_1)

# ----------------------- Validation of Regional and National HVI -----------------------

for i in range(2):
    if False: # Overlaying of the validation data with the regional and national HVI
        shp_overlayer(glob(out_dir_5_1 + "*.shp")[0], hvi_data[i], out_shp_fp[i])

 # Linear class transformation by setting a specific bin size
bins = [0, 10, 18, 26, 34, 42]

for i in range(2):
    if False: # Transforming the HVI classes to the validation classes
        shp_class_transformer(glob(out_shp_fp[i] + "*_overlayed.shp")[0],
                              bins, out_shp_fp[i])

    if False: # Creating the confusion matrix
        confusion_matrix_calculator(glob(out_shp_fp[i] + "*_predclass.shp")[0],
                                    out_csv_fp[i])

    if False: # Calculating the User's and Producer's accuracy
        accuracy_matrix_calculator(glob(out_csv_fp[i]+"*_conf_matrix.csv")[0],
                                   out_csv_fp[i])

# --------------------------------------------------------------------------------------

# Finishing the script
print("Script finished in: {:.2f} s".format(time()-start))
print()