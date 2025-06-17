import glob
from yokogawa_cv8000_utils.functions_legacy import *
from itertools import product
from multiprocessing import Pool

# data folder with raw tif files
folder_data = 'X:/path/of/directory/'

# get measurement files (adjust if less subdirectories)
mlf_files = glob.glob(folder_data + '*/*/*/MeasurementData.mlf')

# folder to save compiled data
folder_export = 'D:/path/output/'
os.makedirs(folder_export, exist_ok=True)


# wrapper
def compile_field_wrapper(plate, well, field, df_imgs, folder_export):
    compile_field(df_imgs, plate=plate, well=well, field=field, folder_export=folder_export)

n_pools = 4

if __name__ == '__main__':
    # parse measurement files and create data frame
    df_imgs, plates, wells, fields = parse_measurement_data(mlf_files)

    # Create a list of tuples containing the arguments for each iteration
    args_list = [(plate, well, field, df_imgs, folder_export) for plate, well, field in product(plates, wells, fields)]

    with Pool(n_pools) as pool:  # specify number of processes
        pool.starmap(compile_field_wrapper, args_list)
