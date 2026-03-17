import argparse
import glob
import os
from multiprocessing import Pool

from .compiling import compile_field
from .parsing import parse_measurement_data


def compile_field_wrapper(args):
    """Wrapper for compile_field to use with multiprocessing.Pool.

    Accepts a tuple of (plate, well, field, df_subset, folder_export, proj_mode)
    where df_subset is already filtered to the relevant (plate, well, field).
    """
    plate, well, field, df_subset, folder_export, proj_mode = args
    compile_field(df_subset, plate=plate, well=well, field=field,
                  folder_export=folder_export, proj_mode=proj_mode)


def main():
    parser = argparse.ArgumentParser(
        description="Compile Yokogawa CV8000 image data into TIFF stacks."
    )
    parser.add_argument("data_folder",
                        help="Path to folder containing raw Yokogawa CV8000 data")
    parser.add_argument("export_folder",
                        help="Path to folder for compiled output")
    parser.add_argument("--depth", type=int, default=3,
                        help="Subdirectory depth to search for MeasurementData.mlf (default: 3)")
    parser.add_argument("--processes", "-p", type=int, default=4,
                        help="Number of parallel processes (default: 4)")
    parser.add_argument("--proj-mode", choices=["mip", "map", "mes"], default="map",
                        help="Projection mode: 'mip' (max intensity projection), 'map' (max average slice), "
                             "or 'mes' (max entropy slice) (default: map)")
    args = parser.parse_args()

    # Build glob pattern based on depth
    wildcard = "/".join(["*"] * args.depth)
    mlf_files = glob.glob(os.path.join(args.data_folder, wildcard, "MeasurementData.mlf"))

    if not mlf_files:
        print(f"No MeasurementData.mlf files found in {args.data_folder}")
        return

    os.makedirs(args.export_folder, exist_ok=True)

    # Parse measurement files and create data frame
    df_imgs, plates, wells, fields = parse_measurement_data(mlf_files)

    # Build task list from actual data groups (skip empty combinations)
    # Each worker receives only the subset it needs, reducing serialization cost
    args_list = []
    for (plate, well, field), df_group in df_imgs.groupby(['plate', 'well', 'field']):
        args_list.append((plate, well, field, df_group, args.export_folder, args.proj_mode))

    print(f"Processing {len(args_list)} (plate, well, field) combinations "
          f"with {args.processes} processes...")

    with Pool(args.processes) as pool:
        pool.map(compile_field_wrapper, args_list)


if __name__ == "__main__":
    main()
