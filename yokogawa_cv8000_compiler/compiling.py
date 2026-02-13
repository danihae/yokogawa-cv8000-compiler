import os

import numpy as np
import tifffile


def compile_field(df_imgs, plate, well, field, folder_export, proj_mode='map'):
    """Compile data for a specific plate, well, and field from a DataFrame of images.

    Parameters:
        df_imgs (DataFrame): DataFrame containing image metadata.
        plate (str): Plate identifier.
        well (str): Well identifier.
        field (int): Field number.
        folder_export (str): Folder path to export the compiled data.
        proj_mode (str, optional): Projection mode. Default is 'map'.
            - 'mip': Maximum Intensity Projection (max across z-slices).
            - 'map': Maximum Average Projection (z-slice with highest mean intensity).

    Returns:
        None

    Raises:
        ValueError: If compiling of data fails or proj_mode is invalid.
    """
    if proj_mode not in ('mip', 'map'):
        raise ValueError(f"Unknown proj_mode '{proj_mode}'. Use 'mip' or 'map'.")

    df_i = df_imgs[(df_imgs['plate'] == plate)
                   & (df_imgs['well'] == well)
                   & (df_imgs['field'] == field)]
    if len(df_i) == 0:
        return

    # Group by action and channel to avoid repeated filtering
    for (action, channel_ij), df_ac in df_i.groupby(['action', 'channel']):
        color_ij = df_ac['color'].iloc[0]
        name_ij = f'plate{plate}_well{well}_field{field}_channel{channel_ij}_color{color_ij}_action{action}'

        data_ij = []

        # Sort once, then group by (begin, timepoint)
        df_ac_sorted = df_ac.sort_values(['begin', 'timepoint', 'z_idx'])
        for (_begin, _tp), df_tp in df_ac_sorted.groupby(['begin', 'timepoint'], sort=True):
            filenames = df_tp['filename'].values
            if len(filenames) == 0:
                continue

            # Read all z-slices for this timepoint at once
            data_ijk = np.stack([tifffile.imread(f) for f in filenames])

            if proj_mode == 'mip':
                data_ij.append(np.max(data_ijk, axis=0))
            else:  # 'map'
                data_ij.append(data_ijk[np.argmax(np.mean(data_ijk, axis=(1, 2)))])

        if not data_ij:
            continue

        data_ij = np.asarray(data_ij)
        pixelsize = float(df_ac['pixelsize'].iloc[0])
        tifffile.imwrite(os.path.join(folder_export, name_ij + '.tif'),
                         data_ij, imagej=True,
                         resolution=(1 / pixelsize, 1 / pixelsize),
                         metadata={'unit': 'um', 'axes': 'ZYX'})
