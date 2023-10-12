import glob
from functions import *
from itertools import product
from multiprocessing import Pool

# data folder
folder_data = 'X:/Lara/2023parakrineAndLiveStains/'

# get measurement files (adjust if less subdirectories)
mlf_files = glob.glob(folder_data + '*/*/*/MeasurementData.mlf')

# parse measurement files and create data frame
df_imgs, plates, wells, fields = parse_measurement_data(mlf_files)

# folder to save compiled data
folder_export = 'D:/Lara/compiled/paracrine/'
os.makedirs(folder_export, exist_ok=True)


# wrapper
def compile_field_wrapper(plate, well, field, df_imgs, folder_export):
    compile_field(df_imgs, plate=plate, well=well, field=field, folder_export=folder_export)


# Create a list of tuples containing the arguments for each iteration
args_list = [(plate, well, field, df_imgs, folder_export) for plate, well, field in product(plates, wells, fields)]

if __name__ == '__main__':
    with Pool(2) as pool:  # specify number of processes
        pool.starmap(compile_field_wrapper, args_list)
