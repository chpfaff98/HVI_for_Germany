from helper_module import *

# This file stores all the functions that are needed for the HVI calculation. It is
# structered in five categories each representing the five consecutive scripts of the
# HVI Program.

# ---------------------------------- 1_grid_creation.py ----------------------------------

def germany_grid_creator(grid_fp:str, out_dir:str):
    '''
    This function creates a 100m grid for the entire Germany
    using the Census 100m bevoelkerungs grid.

    grid_fp: filepath of the bevoelkerungs grid
    out_dir: output directory path
    '''
    # Reading in the grid 
    data = pd.read_csv(grid_fp, sep=";", header=0)

    # Preprocessing
    data = data[data["Einwohner"] != -1] # Removing uninhabited cells
    data = data.rename(columns={"Gitter_ID_100m": "id"}) # Renaming the column

    # Creating the geometry column
    geometry = [Polygon([(x - 50, y - 50), (x + 50, y - 50),(x + 50, y + 50),
                         (x - 50, y + 50)]) for x, y in zip(data["x_mp_100m"], 
                         data["y_mp_100m"])]
    gdf = gpd.GeoDataFrame(data, geometry=geometry)

    # Define CRS & saving it
    gdf.crs = "EPSG:3035" # ETRS89-LAEA

    # Saving the shapefile
    output_shapefile = out_dir + "germany_100m_bevoelkerungs_grid.shp"
    gdf.to_file(output_shapefile)

def raster_mask_creator(shp_fp:str, out_dir:str):
    '''
    This function creates a raster mask from a shapefile.

    shp_fp: filepath of the shapefile to create a raster from
    out_dir: output directory path of the raster file
    '''
    # Reading in the shapefile & adding new mask column
    shp = gpd.read_file(shp_fp)
    
    # Adding a new column
    shp["mask"] = 1

    # Creating the Germany raster grid
    raster = make_geocube(vector_data=shp,
                          measurements=["mask"],
                          resolution=(100,100),
                          fill= None)
    
    # Creating filename
    filename = shp_fp.split("/")[-1]
    filename = filename.replace(".shp", ".tif")

    # Saving the raster
    raster.rio.to_raster(out_dir + filename, driver="GTiff")

def shp_study_area_preprocessor(shp_fp:str, out_dir:str):
    '''
    This function preprocesses the study area shapefile
    from the Geobroker.

    shp_fp: filepath of the shapefile
    out_dir: output directory path
    '''
    # Reading in the shapefile & setting the new epsg
    shp, new_epsg = gpd.read_file(shp_fp), 3035

    # Reprojecting the shapefile
    shp_reproj = shp.to_crs(epsg=new_epsg)

    # Filtering the shapefile
    shp_filtered = shp_reproj[shp_reproj["gem_name"] == "Potsdam"]

    # Saving the shapefile
    shp_filtered.to_file(out_dir + "study_area.shp")

def study_area_buffer_creator(shp_fp:str, buffer_size:int, out_dir:str):
    '''
    This function creates a buffer of the study area shapefile which is 
    important for later clipping.

    shp_fp: filepath of the study area shapefile
    buffer_size: buffer size in meters
    out_dir: output directory path
    '''
    # Reading in the shapefile
    shp = gpd.read_file(shp_fp)

    # Creating the buffer
    shp_buffer = shp.buffer(buffer_size)

    # Saving the shapefile
    shp_buffer.to_file(out_dir + "study_area_{}m_buffer.shp".format(buffer_size))

def study_area_masking(grid_ger_fp:str, study_area_fp:str, out_dir:str):
    '''
    This function masks the Germany 100m grid with the Study Area.

    grid_ger_fp: filepath of the 100m bevoelkerungs grid shapefile
    study_area_fp: filepath of the preprocessed study area
    out_dir: output directory path
    '''
    # Reading in the shapefiles
    grid = gpd.read_file(grid_ger_fp)
    study_area = gpd.read_file(study_area_fp)

    # Masking the grid with the study area
    selected_features = gpd.sjoin(grid, study_area, how="inner", predicate="intersects")

    # Filtering the columns
    column_list = list(selected_features.columns)
    column_list.remove("id")
    column_list.remove("x_mp_100m")
    column_list.remove("y_mp_100m")
    column_list.remove("geometry")
    selected_features = selected_features.drop(column_list, axis=1)
    
    # Saving the study area grid shapefile
    selected_features.to_file(out_dir + "study_area_grid_100m.shp")

# ------------------------------------ 2_regional.py -------------------------------------

def census_filter_sa(grid_fp:str, census_data:str, merkmal:str, auspreaegung_code:int, out_dir:str):
    '''
    This function filters the Census 2011 data by their 
    attributes for the study area and saves them as CSV
    files using the Dask module.

    grid_fp: filepath of the study area 100m grid
    census_data: filepath of the Census 2011 data
    merkmal: Attribute of the Census 2011 data
    auspraegung_code : Attribute ID number
    out_dir: output directory path
    '''
    # Reading in the Grid
    grid = gpd.read_file(grid_fp)

    # Reading the census data of choice
    if census_data == in_dir + "csv_Demographie_100m_Gitter/Bevoelkerung100M.csv":
        df = dd.read_csv(census_data, sep=";", header=0, encoding="latin1")
    else: 
        df = dd.read_csv(census_data, sep=",", header=0, encoding="latin1")

    # Dropping the column id
    df = df.drop(["Gitter_ID_100m_neu"], axis=1)

    # Choosing the grid IDs of the study area
    id_sa = grid["id"].tolist()
    df_id = df[df["Gitter_ID_100m"].isin(id_sa)].compute()

    # Filtering the census data
    df_id_merkmal = df_id[df_id["Merkmal"] == merkmal]
    df_id_merkmal_codeid = df_id_merkmal[df_id_merkmal["Auspraegung_Code"] == auspreaegung_code]

    # Creating the output filename
    out_fn = "{}_{}_SA.csv".format(merkmal, auspreaegung_code) 

    # Saving the filtered data as csv
    df_id_merkmal_codeid.to_csv(out_dir + out_fn, sep=";", encoding="latin1", index=False)

def census_csv_merger(csv_fp:list, out_dir:str):
    '''
    This function merges the filtered csv files into one csv file,
    filters the data and renames the column names.

    csv_fp: list that contains the filepaths of the filtered csv files
    out_dir: output directory path
    '''
    # Starting the loop to merge the filtered csv files
    for i, csv in enumerate(csv_fp):
        # Read in the csv file
        df = pd.read_csv(csv, sep=";", header=0, encoding="latin1")

        # Renaming the id column
        df.rename(columns={"Gitter_ID_100m": "id"}, inplace=True)
        
        # Renaming the Anzahl column based on the filename
        old_fn = csv.split("/")[-1].split(".")[0]
        fn_converter = {"ALTER_KURZ_5_SA":">65a",
                        "ALTER_10JG_1_SA":"<10a",
                        "HHGROESS_KLASS_1_SA":"SH",
                        "STAATSANGE_KURZ_2_SA":"FRG"}
        df.rename(columns={"Anzahl": fn_converter.get(old_fn)}, inplace=True)

        # Dropping the columns that are not needed
        df = df.drop(["Merkmal", "Auspraegung_Code", "Auspraegung_Text", "Anzahl_q"], axis=1)

        # Merging the dataframes into one dataframe
        if i == 0:
            df_merged = df
        else:
            df_merged = df_merged.merge(df, on="id", how="outer")

        # After the last loop, the merged dataframe is saved as csv 
        if i == len(csv_fp)-1:
            df_merged.to_csv(out_dir + "census_merged.csv", sep=";", index=False)

def shp_creator(csv_fp:str, grid_fp:str, merge_method:str, out_dir:str):
    '''
    This function creates a shapefile for the Census 2011 csv data for visualization
    with the 100m grid.

    csv_fp: filepath of the filtered csv data
    grid_fp: filepath of the grid
    merge_method : method of the spatial join like "left" or "inner"
    out_dir: output directory path
    '''
    # Importing the warnings module
    import warnings
    warnings.filterwarnings("ignore")

    # Reading the csv file
    df = pd.read_csv(csv_fp, sep=";", header=0, encoding="latin1")
    df = df.rename(columns={"Gitter_ID_100m":"id"})

    # Built-in error message
    c1 = ["id", "Merkmal", "Auspraegung_Code", "Auspraegung_Text"]
    c2 = [len(df), 1, 1, 1]
    for i in c1:
        if len(df[i].unique()) == c2[c1.index(i)]:
            pass
        else:
            print("ERROR Message: {} is not correctly filtered!".format(i))
            sys.exit()

    # Reading the grid and merging 
    grid = gpd.read_file(grid_fp)
    joined_grid = grid.merge(df, on="id", how=merge_method)

    # Creating filename
    filename = csv_fp.split("/")[-1]
    filename = filename.replace(".csv", ".shp")

    # Saving the shapefile
    joined_grid.to_file(out_dir+filename, driver="ESRI Shapefile")

def tif_reprojector(tif_fp:str, out_dir:str):
    '''
    This function reprojects the raster data to EPSG:3035.

    tif_fp: filepath of the raster file
    out_dir: output directory path
    '''
    # Reading in the data
    tif, wkt = gdal.Open(tif_fp), gdal.Open(tif_fp).GetProjection()

    # Detecting the current projection using the WKT file
    s, e = wkt.find('"')+1, wkt.find(',')-1
    wkt_proj = wkt[s:e]

    # Printing out the current projection
    print("File: ", tif_fp.split("/")[-1])
    print("Corresponding projection: {}".format(wkt_proj))
    print()

    # Checking the data projection
    if wkt_proj == "ETRS89_LAEA_Europe":
        # Copying the raster data to the output directory
        shutil.copy(tif_fp, out_dir)

    elif wkt_proj == "ETRS89-extended / LAEA Europe":
        # Copying the raster data to the output directory
        shutil.copy(tif_fp, out_dir)

    # Reprojecting the raster data to EPSG:3035
    else:
        # Creating the output filename
        filename = tif_fp.split("/")[-1].replace(".tif", "_epsg3035.tif")
        filename_fp = out_dir+filename

        # Reprojecting the raster data
        gdal.Warp(filename_fp, tif, dstSRS="EPSG:3035",
                  creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"],
                  outputType=gdal.GDT_Float32)
        tif = None

def tif_clipper(tif_fp:str, shp_fp:str, out_dir:str):
    '''
    This function clips the raster data to the study area.

    tif_fp: filepath of the raster file 
    shp_fp: filepath of the buffer shapefile to clip 
    out_dir: folderpath of the output file 
    '''
    # Reading in the data
    tif = gdal.Open(tif_fp) 
    
    # Creating filename and output filepath
    filename = tif_fp.split("/")[-1]
    filename = filename.replace(".tif", "_clipped.tif")
    out_fp = out_dir + filename

    # Clipping the raster data
    gdal.Warp(out_fp, tif, cutlineDSName=shp_fp, cropToCutline=True, dstNodata=np.nan,
              creationOptions=["COMPRESS=LZW", "BIGTIFF=YES"])
    tif = None

def tif_resampler_regional(tif1_fp:str, tif2_fp:str, resampling_method, out_dir:str):
    '''
    This function resamples a raster to the spatial resolution and 
    extent of a reference raster.

    tif1_fp: filepath of the source raster file
    tif2_fp: filepath of the mask raster file
    resampling_method: resampling method from Rasterio
    out_dir: output directory path of the resampled raster
    '''
    # Opening the source raster
    with rio.open(tif1_fp) as src:
        src_profile = src.profile

        # Open the reference raster
        with rio.open(tif2_fp) as ref:
            # Read metadata
            ref_profile = ref.profile

            # Creating empty array
            data = src.read(1).astype("float32")
            dst_data = np.empty((ref_profile["height"],
                                 ref_profile["width"]), 
                                 dtype=np.float32)

            # Reprojecting the raster
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=resampling_method
            )

            # Update the metadata
            src_profile.update({
                "width": ref_profile["width"],
                "height": ref_profile["height"],
                "transform": ref_profile["transform"],
                "crs": ref_profile["crs"],
                "dtype": "float32"
            })

        # Creating the output filename
        filename = tif1_fp.split("/")[-1].replace(".tif", "_resampled.tif")

        # Write the resampled raster to a new file
        with rio.open(out_dir + filename, "w", **src_profile) as dst:
            dst.write(dst_data, 1)

def tif_to_shp_joiner(tif_fp:str, grid_fp:str, out_dir:str):
    '''
    This function extracts and joins the resampled raster values to the 100m study area
    grid shapefile based on their geometry. Then this modified shapefile is saved as a 
    new shapefile.

    tif_fp: filepath of the resampled raster file
    grid_fp: filepath of the 100m study area grid shapefile
    out_dir: output directory path
    '''
    # open the data
    tif, shp = rio.open(tif_fp), gpd.read_file(grid_fp)

    # Loop through each geometry in the shapefile
    for index, row in shp.iterrows():
        # Extract geometry and transform it to GeoJSON format
        geom = row.geometry.__geo_interface__ 
        
        # Extract raster values using the geometry
        out_image, _ = mask(tif, [geom], crop=True) 
        
        # Getting the pixel values
        tif_pixel_value = out_image[0]

        # Adding the pixel value to the location in the shapefile
        shp.at[index, "Mean"] = tif_pixel_value 
    tif.close()

    # Creating the filename
    filename = tif_fp.split("/")[-1].replace(".tif", ".shp")

    # Saving the shapefile
    shp.to_file(out_dir + filename)

def shp_res_csv_saver(shp_fp:str, out_dir:str):
    '''
    This function saves the resampled shapefiles as csv file.

    shp_fp: filepath of the shapefile
    out_dir: output directory path
    '''
    # Reading in the shapefile
    shp = gpd.read_file(shp_fp)

    # Read the columns "Mean" as float
    shp["Mean"] = shp["Mean"].astype(float)

    # Renaming and dropping the column names
    fn_converter = {"Potsdam_2022_canopy_cover_fraction_10m_FORCE_clipped_resampled" : "CP",
                    "potsdam_vh_2022_10m_force_clipped_resampled" : "VH",
                    "potsdam_gv_rb_2022_10m_force_clipped_resampled" : "GV",
                    "potsdam_versiegelung_maerz_2022_rf_50cm_epsg3035_clipped_resampled" : "SEA",
                    "LST_BRD_median_gapFilled_2018_2023_clip_epsg3035_clipped_resampled" : "LST"}
    old_fn = shp_fp.split("/")[-1].split(".")[0]
    shp = shp.rename(columns={"Mean": fn_converter.get(old_fn)})
    shp = shp.drop(columns=["geometry"])

    # Creating the output filename
    filename = shp_fp.split("/")[-1].replace(".shp", ".csv")

    # Saving the shapefile as csv
    shp.to_csv(out_dir+filename, sep=";", index=False)

def df_merger(csv_fp:list, census_merged_csv_fp:str, out_dir:str, preprocess=True, print_head=True):
    '''
    This function merges the census csv data and the raster csv files
    on the study area grid based on the ID number.

    csv_fp: list of the filepaths of the raster csv files
    census_merged_csv_fp: filepath of the merged census csv file
    out_dir: output directory path
    preprocess: if True, the raster data will be preprocessed
    print_head: if True, the head of the merged dataframe will be printed
    '''
    # Reading in the merged census csv data
    hvi_df = pd.read_csv(census_merged_csv_fp, sep=";", header=0, encoding="latin1")

    # Merging the census data with the indicator csv files
    for i in range(len(csv_fp)):
        # Reading the raster csv files 
        df_indicator = pd.read_csv(csv_fp[i], sep=";", header=0, encoding="latin1")

        # Dorpping x and y columns
        df_indicator.drop(columns=["y_mp_100m", "x_mp_100m"], inplace=True)

        # Merging the dataframes
        hvi_df = hvi_df.merge(df_indicator, on="id", how="outer")

    # Correcting and preprocessing the indicator columns
    if preprocess == True:
        # Preprocessing the indicator columns
        hvi_df["GV"] = hvi_df["GV"] * -1
        hvi_df["CP"] = hvi_df["CP"] * -1
        hvi_df["VH"] = hvi_df["VH"] * -1
        hvi_df["SEA"] = hvi_df["SEA"] - 1

    # Printing the head of dataframe
    if print_head == True:
        print(hvi_df.head())

    # Saving the merged dataframe as csv 
    hvi_df.to_csv(out_dir + "df.csv", sep=";", index=False)

def spearman_correlation_analysis_sa(df_fp:str, out_dir:str):
    '''
    This function calculates the correlation matrix of the 
    variables of the study area and saves it as a png file.

    df_fp: filepath of the merged dataframe
    out_dir: output directory path
    '''
    # Reading in the important columns of the dataframe
    df = pd.read_csv(df_fp, sep=";", header=0, encoding="latin1").drop(["id"], axis=1)

    # Calculating the correlation matrix and processing it
    corr = df.corr(method="spearman")

    # Settings before the Plot
    labels = list(corr.columns)
    sns.set(style="white")

    # Creating a mask for upper triangle (better visualization)
    mask = np.zeros_like(corr, dtype=bool)
    mask[np.triu_indices_from(mask)] = True
    corr[mask] = np.nan

    # Creating the correlation matrix
    fig, ax = plt.subplots(figsize=(10,10))
    sns.heatmap(corr, annot = True, cmap="YlOrBr", vmin=0, vmax=1, ax=ax)

    # Setting the colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label(r"Spearman Correlation Coefficient - $r_{sp}$", fontsize=14)

    # Processing the axes
    ax.set_xticklabels(labels[:-1]+[""])
    ax.set_yticklabels([""]+labels[1:])

    # Saving the correlation matrix
    # plt.show()
    fig.savefig(out_dir + "12_SPRCM_Potsdam.png", dpi=300)

def dataframe_column_dropper(df_fp:str, drop_col:list, out_dir:str, print_head=True):
    '''
    This functions drops columns of a dataframe and saves it.

    df_fp: filepath of the dataframe
    drop_col: list of the columns to drop from the dataframe
    out_dir: output directory path
    print_head: if True, the head of the dataframe will be printed
    '''
    # Reading in the dataframe
    df = pd.read_csv(df_fp, sep=";", header=0, encoding="latin1")
    
    # Dropping the columns
    df.drop(columns=drop_col, inplace=True)

    # Pint the head of the dataframe
    if print_head == True:
        print(df.head())

    # Filename creation
    filename = df_fp.split("/")[-1].replace(".csv", "_uncorrelated.csv")

    # Saving the dataframe
    df.to_csv(out_dir + filename, sep=";", index=False)

def mean_std_sa_calculator(df_fp:str, out_dir:str, print_df=True):
    '''
    This function calculates the mean and std of the variables for the study area.

    df_fp: filepath of the merged dataframe
    out_dir: output directory path
    print_df: if True, the dataframe will be printed
    '''
    # Reading in the dataframe
    df = pd.read_csv(df_fp, sep=";", header=0, encoding="latin1")

    # Dropping the id column 
    cols = list(df.columns)
    cols.remove("id")

    # Creating an empty dataframe
    rows = ["mean", "std"]
    mean_std_df = pd.DataFrame(index=rows, columns=cols)
    mean_std_df.index.name = "stat"

    # Calculating the mean and std
    for i in cols:
        mean_std_df.loc["mean", i] = np.round(np.nanmean(df[i]),4)
        mean_std_df.loc["std", i] = np.round(np.nanstd(df[i]),4)

    # Printing mean and std dataframe
    if print_df == True:
        print(mean_std_df)

    # Saving the dataframe
    mean_std_df.to_csv(out_dir + "mean_std_sa.csv", sep=";", index=True)

# ------------------------------------ 3_national.py ------------------------------------

def census_filter_ger(census_data:str, merkmal:str, auspreaegung_code:int, out_dir:str):
    '''
    This function filters the Census 2011 data by 
    their attributes for entire Germany.

    census_data: filepath of the Census 2011 data
    merkmal: Attribute of the Census 2011 data
    auspraegung_code : Attribute ID number
    out_dir: output directory path
    '''
    # Reading the census data of choice
    if census_data == in_dir + "csv_Demographie_100m_Gitter/Bevoelkerung100M.csv":
        df = dd.read_csv(census_data, sep=";", header=0, encoding="latin1")
    else: 
        df = dd.read_csv(census_data, sep=",", header=0, encoding="latin1")

    # Dropping the column id
    df = df.drop(["Gitter_ID_100m_neu"], axis=1)

    # Filtering the census data
    df_id_merkmal = df[df["Merkmal"] == merkmal]
    df_id_merkmal_codeid = df_id_merkmal[df_id_merkmal["Auspraegung_Code"] == auspreaegung_code]

    # Creating the output filename
    out_fn = "{}_{}_GER.csv".format(merkmal, auspreaegung_code) 

    # Saving the filtered data as csv
    df_id_merkmal_codeid.compute().to_csv(out_dir + out_fn, sep=";", encoding="latin1", index=False)

def tif_extent_cropper(tif1_fp:str, tif2_fp:str, out_dir:str):
    '''
    This function crops a raster to the spatial extent of another raster.

    tif1_fp: filepath of the raster that will be cropped
    tif2_fp: filepath of the raster with correct extent
    out_dir: output directory path of the cropped raster
    '''
    # Reading in the raster with the desired extent
    with rio.open(tif2_fp) as r2:
        bbox = r2.bounds
    
    # Reading in the raster to crop and creating geometry cropping mask
    with rio.open(tif1_fp) as r1:
        geometry = [{"type": "Polygon",
                    "coordinates": [[[bbox.left, bbox.bottom],
                                     [bbox.left, bbox.top], 
                                     [bbox.right, bbox.top],
                                     [bbox.right, bbox.bottom],
                                     [bbox.left, bbox.bottom]]]}]
    
        # Cropping the raster
        cropped_raster, transformation = mask(r1, geometry, crop=True)
        metadata = r1.meta

        # Updating the metadata
        metadata.update({"driver": "GTiff",
                        "height": cropped_raster.shape[1],
                        "width": cropped_raster.shape[2],
                        "transform": transformation})

        # Creating the output filename
        filename = tif1_fp.split("/")[-1].replace(".tif", "_cropped.tif")

        # Saving the cropped raster
        with rio.open(out_dir + filename, "w", **metadata) as r3:
            r3.write(cropped_raster)

def tif_filler(tif1_fp:str, tif1_nan:int, tif2_fp:str, out_dir:str):
    '''
    This function fills the NaN values of raster with the corresponding values 
    of a second raster. IMPORTANT: The raster need the same extent, resolution 
    and projection.

    tif1_fp: filepath of the raster to be filled
    tif1_nan: nan value of the raster to be filled
    tif2_fp: filepath of the raster with the filler values
    out_dir: output directory path of the filled raster
    '''
    # Reading in the rasters
    r1, r2 = gdal.Open(tif1_fp), gdal.Open(tif2_fp)
    b1, b2 = r1.GetRasterBand(1), r2.GetRasterBand(1)

    # Reading the data as arrays
    data1, data2 = b1.ReadAsArray(), b2.ReadAsArray()

    # Replacing the nan values in the raster 1 with values from raster 2
    data1 = np.where(data1 == tif1_nan, data2, data1)

    # Creating the output filename
    filename = tif1_fp.split("/")[-1].replace(".tif", "_filled.tif")

    # Creating a new GeoTIFF file and setting the metadata
    driver = gdal.GetDriverByName("GTiff")
    r3 = driver.Create(out_dir + filename, r1.RasterXSize, r1.RasterYSize, 1, b1.DataType)
    r3.GetRasterBand(1).WriteArray(data1) # writing the data in the raster
    r3.GetRasterBand(1).SetNoDataValue(tif1_nan) # setting the nan value
    r3.SetProjection(r1.GetProjection()) # setting the projection
    r3.SetGeoTransform(r1.GetGeoTransform()) # setting the geotransform

    # Closing the rasters
    r1, r2, r3 = None, None, None

def tif_edt_filler(tif_fp:str, out_dir:str, split=True, fill=True, merging=True, delete=False):
    '''
    This function fills the NaN values of a raster with the nearest neighbor values and it
    was specifically designed for very large raster files. First this function splits the 
    raster into several subrasters. Then the distance of each nodata pixel to its nearest
    neighbor and their corresponding indices are calculated. Then the nodata values are filled
    with their nearest neighbor values. Finally each raster subset is merged together into one
    final raster.
    
    IMPORTANT: During the process a temporary folder is created in the output directory, BUT
    will be deleted after the final raster was computed.

    tif_fp: filepath of the raster to be filled
    out_dir: output directory path of the filled raster
    split: if True, the raster will be splitted into 100 subrasters
    fill: if True, the splitted raster will be filled with the nearest neighbor values
    merging: if True, the filled raster will be merged together
    delete: if True, the temporary folder will be deleted
    '''

    # Creating the temporary directories
    temp_folders = [out_dir + "tif_nb_filler_temp/tifs_splitted/", 
                    out_dir + "tif_nb_filler_temp/tifs_filled/"]
    for i in temp_folders:
        os.makedirs(i, exist_ok=True)
    
    # Defining nested functions
    def raster_splitter(tif_fp:str):
        # Reading in the raster
        r = gdal.Open(tif_fp)

        # Getting the raster size
        x, y = r.RasterXSize, r.RasterYSize

        # Calculating tile size
        tile_x, tile_y = x // 10, y // 10
        
        # Looping through the raster (100 subrasters)
        for i in range(10):
            for j in range(10):
                # window calculation 
                x_coord, y_coord = i * tile_x, j * tile_y 

                # Read the data
                ds = r.ReadAsArray(x_coord, y_coord, tile_x, tile_y)

                # New raster creation
                driver = gdal.GetDriverByName("GTiff")
                r_subset_fn =  temp_folders[0] + f"r_subset_{i}_{j}.tif"
                r_subset = driver.Create(r_subset_fn, tile_x, tile_y, 1, r.GetRasterBand(1).DataType)
                r_subset.SetGeoTransform((r.GetGeoTransform()[0] + x_coord * r.GetGeoTransform()[1],
                                        r.GetGeoTransform()[1],
                                        0,
                                        r.GetGeoTransform()[3] + y_coord * r.GetGeoTransform()[5],
                                        0, 
                                        r.GetGeoTransform()[5]))
                r_subset.SetProjection(r.GetProjection())
                r_subset.GetRasterBand(1).WriteArray(ds)
                r_subset = None
        r = None

    def raster_edn_replacer(tif_fp:str):
        with rio.open(tif_fp) as r:
            # Reading in the raster & setting no data value
            data = r.read(1)
            mask = data == 128

            # Compute the nearest neighbor indices
            indices = distance_transform_edt(mask, return_distances=False, return_indices=True)

            # Pulling the nearest neighbor values to the location 
            # of the nodata values (based on indices)
            data_filled = data[tuple(indices)]

            # Filename creation
            filename = tif_fp.split("/")[-1].replace(".tif", "_filled.tif")

            # Save the filled raster
            with rio.open(temp_folders[1] + filename, "w", **r.profile) as r_filled:
                r_filled.write(data_filled, 1)

    def raster_folder_merger(tif_folder:str):
        # Creating lists 
        rasters, data_list = glob(tif_folder + "*.tif"), []

        # Looping through the rasters
        for fp in rasters:
            data = rio.open(fp)
            data_list.append(data)
        merged_raster, out_transform = merge(data_list)
        metadata = data_list[0].meta.copy()

        # Update the metadata with the merged data's dimensions and transform
        metadata.update({"driver": "GTiff",
                         "height": merged_raster.shape[1],
                         "width": merged_raster.shape[2],
                         "transform": out_transform})

        # Write the merged data to a new file
        filename = "Versiegelung_RF_Regression_Deutschland_multitemp_2023_cropped_filled_nb_filled.tif"
        with rio.open(out_dir + filename, "w", **metadata) as final_raster_merged:
            final_raster_merged.write(merged_raster)

        # Closing all opened rasters
        for raster in data_list:
            raster.close()

    # Workflow Execution
    if split == True:
        raster_splitter(tif_fp)

    if fill == True:
        fp = glob(temp_folders[0] + "*.tif")
        for i in fp:
            raster_edn_replacer(i)

    if merging == True:
        raster_folder_merger(temp_folders[1])

    if delete == True:
        shutil.rmtree(out_dir + "tif_nb_filler_temp/")

def tif_resampler_national(tif1_fp:str, tif2_fp:str, nodata, method, out_dir:str, nodata_flag=False):
    '''
    This function reprojects a raster to the spatial extent of another raster.

    tif1_fp: filepath of the source raster file
    tif2_fp: filepath of the mask raster file
    nodata: nodata value of the raster
    method: resampling method from Rasterio
    out_dir: output directory path of the resampled raster
    nodata_flag: if True, the nodata values will be set to nodata value
    '''
    # Reading in the source raster
    with rio.open(tif1_fp) as src:
        src_profile = src.profile

        # Open the reference raster
        with rio.open(tif2_fp) as ref:
            # Read metadata
            ref_profile = ref.profile

            # Creating empty array
            data = src.read(1).astype("float32")

            # set nan values to np.nan
            if nodata_flag == True:
                data[data == nodata] = np.nan

            # Creating empty array
            dst_data = np.empty((ref_profile["height"], ref_profile["width"]), dtype=np.float32)

            # Reprojecting the raster
            reproject(
                source=data,
                destination=dst_data,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=ref.transform,
                dst_crs=ref.crs,
                resampling=method
            )

            # Update the metadata
            src_profile.update({
                "width": ref_profile["width"],
                "height": ref_profile["height"],
                "transform": ref_profile["transform"],
                "crs": ref_profile["crs"],
                "dtype": "float32"
            })

        # Creating the output filename
        filename = tif1_fp.split("/")[-1].replace(".tif", "_100m.tif")

        # Write the resampled raster to a new file
        with rio.open(out_dir + filename, "w", **src_profile) as r:
            r.write(dst_data, 1)

def tif_masker(tif_fp:str, mask_tif_fp:str, out_dir:str):
    '''
    This function masks the source raster with the mask raster.

    tif_fp: filepath of the raster file
    mask_tif_fp: filepath of the mask raster file
    out_dir: output directory path
    '''
    # Reading in the two raster files
    with rio.open(mask_tif_fp) as r1, rio.open(tif_fp) as r2:
        
        # Reading in the first band of the rasters
        raster1 = r1.read(1)
        raster2 = r2.read(1)

        # Multiply the rasters
        result = raster1 * raster2

        # Update the metadata
        metadata = r1.meta
        metadata.update(count=1)

        # Creating the output filename
        filename = tif_fp.split("/")[-1].replace(".tif", "_masked.tif")

        # Save the masked raster
        with rio.open(out_dir + filename, "w", **metadata) as r:
            r.write(result, 1)

def mean_std_ger_calculator(out_dir:str):
    '''
    This function calculates the mean and std of the variables for the study area.
    It automatically takes refers to all folders in the output directory. Input
    files don't need to be specified, because the function will automatically read
    in the files as they are defined below.

    out_dir: output directory path
    '''
    # Data filepaths for the entire Germany data
    census = [out_dir_3_1+"ALTER_KURZ_5_GER.csv", 
              out_dir_3_1+"ALTER_10JG_1_GER.csv",
              out_dir_3_1+"HHGROESS_KLASS_1_GER.csv",
              out_dir_3_1+"STAATSANGE_KURZ_2_GER.csv"]
    raster = [glob(out_dir_3_5+"GV_*_masked.tif")[0],
              glob(out_dir_3_5+"Versiegelung_*_masked.tif")[0],
              glob(out_dir_3_5+"LST_*_masked.tif")[0]]

    # Creating an empty dataframe
    census_cols, raster_cols = [">65a", "<10a", "SH", "FRG"], ["GV", "SEA", "LST"]
    cols = census_cols + raster_cols
    rows = ["mean", "std"]
    mean_std_df = pd.DataFrame(index=rows, columns=cols)
    mean_std_df.index.name = "stat"

    # Calculating the mean and std for the census data
    for i, column in enumerate(census_cols):
        df = pd.read_csv(census[i], sep=";", header=0, encoding="latin1")
        mean_std_df.loc["mean", column] = np.round(np.nanmean(df["Anzahl"]), 4)
        mean_std_df.loc["std", column] = np.round(np.nanstd(df["Anzahl"]), 4)

    # Calculating the mean and std for the raster data
    for i, column in enumerate(raster_cols):
        tif = rio.open(raster[i])
        if column == "SEA":
            mean_std_df.loc["mean", column] = np.nanmean(tif.read(1))
            mean_std_df.loc["std", column] = np.nanstd(tif.read(1))
        else:
            mean_std_df.loc["mean", column] = np.round(np.nanmean(tif.read(1)), 4)
            mean_std_df.loc["std", column] = np.round(np.nanstd(tif.read(1)), 4)

    # Postprocessing the dataframe
    mean_std_df.loc["mean", "GV"] = mean_std_df.loc["mean", "GV"] * -1 # Green Volume
    mean_std_df["SEA"] = mean_std_df["SEA"] * 0.01 # Sealing
    for i in rows:
        mean_std_df.loc[i, "SEA"] = np.round(mean_std_df.loc[i, "SEA"], 4)

    # Saving the dataframe
    mean_std_df.to_csv(out_dir + "mean_std_ger.csv", sep=";", index=True)

# ------------------------------------ 4_hvi_builder.py ------------------------------------

def hvi_calculator(df_fp:str, stat_fp:str, out_dir:str):
    '''
    This function calculates the HVI with the mean and std of spefific area
    and saves it as a shapefile & csv file.

    df_fp: filepath of the merged dataframe
    stat_fp: filepath of the mean and std dataframe
    out_dir: output directory path
    '''
    # Reading in the merged dataframe and statistics
    df = pd.read_csv(df_fp, sep=";", header=0, encoding="latin1")
    stat_area = pd.read_csv(stat_fp, sep=";", header=0, index_col="stat")

    # Classification intervals
    bins = [float("-inf"), -2, -1, 0, 1, 2, float("inf")]
    labels = [1, 2, 3, 4, 5, 6]

    # Settings variables
    cols = list(df.columns)[1:] # without the id column
    hvi_name = []

    # Z-score calculation
    for i in cols:
        mean = stat_area.loc["mean", i]
        std = stat_area.loc["std", i]
        zs = np.round((df[i] - mean) / std, 4)
        df["zs_"+i] = zs

    # HVI calculation
    for i in cols:
        df["hvi_"+i] = pd.to_numeric(pd.cut(df["zs_"+i], bins=bins, labels=labels, right=True))
        hvi_name+=["hvi_"+i]

    # Final HVI calculation and saving as csv
    df["HVI_sum"] = df[hvi_name].sum(skipna = True, axis=1)

    # Filtering out the area name
    area_name = stat_fp.split("/")[-1].split(".")[0].split("_")[-1]
    
    # Saving the HVI as csv
    df.to_csv(out_dir + "FINAL_HVI_{}.csv".format(area_name), sep=";", index=False)
    
    # Saving the HVI as shapefile
    grid = gpd.read_file(out_dir_1_2 + "study_area_grid_100m.shp")
    grid_hvi = grid.merge(df, on="id", how="left")
    grid_hvi.to_file(out_dir + "FINAL_HVI_{}.shp".format(area_name), driver="ESRI Shapefile")

def shp_hvi_diff(shp1_fp:str, shp2_fp:str, out_dir:str):
    '''
    This function calculates the difference between two Shapefiles
    with the same resolution.

    shp1_fp: filepath of the first shapefile
    shp2_fp: filepath of the second shapefile
    out_dir: output directory path
    '''
    # deactivating the warnings
    import warnings
    warnings.filterwarnings("ignore")

    # Reading in the data
    shp1, shp2 = gpd.read_file(shp1_fp), gpd.read_file(shp2_fp)

    # Renaming the HVI_sum column
    shp1 = shp1.rename(columns={"HVI_sum": "HVI_sum_sa"})
    shp2 = shp2.rename(columns={"HVI_sum": "HVI_sum_ger"})

    # Drop all columns but not HVI sum and geometry
    shp1 = shp1[["id", "HVI_sum_sa", "geometry"]]
    shp2 = shp2[["id", "HVI_sum_ger", "geometry"]]

    # Reading in the first shapefile & dropping the geometry
    data = shp1.drop(["geometry"], axis=1)

    # Merging with the other shapefile
    shp_merged = shp2.merge(data, on='id', how='left')

    # Calculating the difference between the Hvi columns
    shp_merged["hvi_diff"] = shp_merged["HVI_sum_ger"] - shp_merged["HVI_sum_sa"]
    
    # Saving the shapefile
    shp_merged.to_file(out_dir + "hvi_diff.shp", driver="ESRI Shapefile")

# ------------------------------------ 5_validation.py ------------------------------------

def shp_validation_preprocessor(shp_fp:str, out_dir:str):
    '''
    This function preprocesses the validation shapefile. It reprojects it and
    filters the correct data from the attribute table.

    shp_fp: filepath of the shapefile
    out_dir: output directory path
    '''
    # Reading in the shapefile & setting the new epsg
    shp, new_epsg = gpd.read_file(shp_fp), 3035

    # Reprojecting the shapefile & saving it
    shp_reproj = shp.to_crs(epsg=new_epsg)

    # Filtering the shapefile (Removing & filtering columns)
    shp_filtered = shp_reproj[["PET_KLASSE", "geometry"]]
    shp_filtered = shp_filtered[shp_filtered["PET_KLASSE"] != "keine Bewertung"]

    # Saving the shapefile
    shp_filtered.to_file(out_dir + "val_data.shp")

def shp_overlayer(shp_val_fp:str, shp_hvi_fp:str, out_dir:str):
    '''
    This function cuts the validation shapefile onto the 100m grid cells 
    of the HVI shapefile.

    shp_val_fp: filepath of the validation shapefile
    shp_hvi_fp: filepath of the HVI shapefile
    out_dir: output directory path
    '''
    # Reading in both shapefiles
    shp_val, shp_hvi = gpd.read_file(shp_val_fp), gpd.read_file(shp_hvi_fp)

    # Cutting the validation shapefile onto the HVI shapefile
    shp_val_cut = gpd.overlay(shp_val, shp_hvi, how="intersection")

    # Remove all columns except "PET_KLASSE" and "geometry" and "HVI_sum"
    shp_val_cut = shp_val_cut[["PET_KLASSE", "HVI_sum", "geometry"]]

    # Filename creations
    filename_1 = shp_val_fp.split("/")[-1].replace(".shp", "_")
    filename_2 = shp_hvi_fp.split("/")[-1].split(".")[-2].split("_")[-1]

    # Saving the shapefile
    shp_val_cut.to_file(out_dir + filename_1 + filename_2 + "_overlayed.shp")

def shp_class_transformer(shp_fp:str, bins:list, out_dir:str):
    '''
    This class transforms the HVI values into the validation
    classes and saves it as a shapefile.

    shp_fp: filepath of the shapefile
    bins: list of the classification intervals
    out_dir: output directory path
    '''
    # Reading in the shapefile
    shp = gpd.read_file(shp_fp)

    # classification transformation
    # Ordered like the bins!
    labels = ["schwache Belastung",
              "maessige Belastung",
              "starke Belastung",
              "sehr starke Belastung",
              "extreme Belastung"]
    
    # Performing the classification transformation
    shp["Pred_class"]=pd.cut(shp["HVI_sum"], bins=bins, labels=labels, right=True).astype(str)

    # filename creation
    filename = shp_fp.split("/")[-1].replace(".shp", "_predclass.shp")

    # Saving the shapefile
    shp.to_file(out_dir + filename)

def confusion_matrix_calculator(shp_fp:str, out_dir:str):
    '''
    This function creates a confusion matrix from the validation shapefile.

    shp_fp: filepath of the shapefile
    out_dir: output directory path
    '''
    # Reading in the shapefile
    shp = gpd.read_file(shp_fp)

    # Filter out the GER
    area = shp_fp.split("/")[-1].split(".")[0].split("_")[-3]
    if area == "SA": area = "reg"
    if area == "GER": area = "nat"

    # Creating the confusion matrix
    # Index are values to group by in the rows.
    # Columns are values to group by in the columns.
    conf_matrix = pd.crosstab(index = shp["Pred_class"], # rows (x-axis)
                              columns = shp["PET_KLASSE"], # columns (y-axis) -> Validation data
                              rownames=["Predicted (by {} HVI)".format(area)])

    # Sum all values rowwise and columnwise
    conf_matrix["Total"] = conf_matrix.sum(axis=1)
    conf_matrix.loc["Total"] = conf_matrix.sum(axis=0)

    # Reordering the rows and columns
    new_order = ["schwache Belastung",
                 "maessige Belastung",
                 "starke Belastung",
                 "sehr starke Belastung",
                 "extreme Belastung",
                 "Total"]
    conf_matrix = conf_matrix.reindex(index=new_order)
    conf_matrix = conf_matrix.reindex(columns=new_order)

    # Filename creation
    filename = shp_fp.split("/")[-1].replace(".shp", "_conf_matrix.csv")

    # Saving the confusion matrix
    conf_matrix.to_csv(out_dir + filename, sep=";")

def accuracy_matrix_calculator(csv_fp:str, out_dir:dir):
    '''
    This function computes different error metrics from the confusion matrix.

    csv_fp: filepath of the confusion matrix csv file
    out_dir: output directory path
    '''
    # Reading in the confusion matrix
    conf_matrix = pd.read_csv(csv_fp, sep=";", header=0, index_col=0)
    
    # Preprocessing the row names
    row_names = list(conf_matrix.index)
    row_names[-1] = "all_classes"
    
    # Create a error matrix pandas dataframe with the index names are the rownames from the confusion matrix
    error_matrix = pd.DataFrame(index = row_names, columns = ["Users_acc_%", "Prod_acc_%", "Overall_acc_%"])
    error_matrix.index.name = "Classes"

    # Starting to fill the empty error matrix with the statistics
    sum_dia = 0
    for i in range(5):
        # User's accuracy
        error_matrix.iloc[i,0] = np.round(conf_matrix.iloc[i,i] / conf_matrix.iloc[i,5] * 100,2)
        # Producer's accuracy
        error_matrix.iloc[i,1] = np.round(conf_matrix.iloc[i,i] / conf_matrix.iloc[5,i] * 100,2)
        # For later overall accuracy calculation
        sum_dia += conf_matrix.iloc[i,i]

    # Overall accuracy
    error_matrix.iloc[5,2] = np.round(sum_dia / conf_matrix.iloc[5,5] * 100,2)
    
    # Print the error matrix
    print(error_matrix)

    # Filename
    filename = csv_fp.split("/")[-1].split("_")[2]

    # Save the error matrix
    error_matrix.to_csv(out_dir + "error_matrix_" + filename + ".csv", sep=";")