import os
import re
import string

import numpy as np
import pandas as pd
import tifffile
import xmltodict


def parse_measurement_data(mlf_files):
    """create dataframe with file meta data"""

    meta_imgs = []

    for mlf_file in mlf_files:
        # read MeasurementData.mlf file with single tif-file information
        with open(mlf_file) as fd:
            doc = xmltodict.parse(fd.read(),
                                  process_namespaces=True,
                                  namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},
                                  attr_prefix="", cdata_key="Value"
                                  )
        dirname = os.path.dirname(mlf_file) + '/'
        list_record = doc['MeasurementData']['MeasurementRecord']
        # get conditions from Wells.txt file
        infotxt = os.path.dirname(os.path.dirname(mlf_file)) + '/Wells.txt'
        if os.path.exists(infotxt):
            with open(infotxt) as fd:
                data_str = fd.read()
                data_dict = {}
                lines = data_str.strip().split('\n')

                for line in lines:
                    if line.strip():  # Check if the line is not empty after stripping whitespace
                        key, value = line.split(' - ')
                        data_dict[key] = value
        else:
            data_dict = {}
        # read MeasurementDetail.mrf file with metadata
        mrf_file = dirname + 'MeasurementDetail.mrf'
        with open(mrf_file, encoding="utf-8-sig") as fd:
            xml_data = fd.read()
            doc = xmltodict.parse(xml_data,
                                  process_namespaces = True,
                                  namespaces = {"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},
                                  attr_prefix = "", cdata_key = "Value"
                                  )
            Title = doc['MeasurementDetail']['Title']
            BeginTime = pd.to_datetime(doc['MeasurementDetail']['BeginTime'])
            EndTime = pd.to_datetime(doc['MeasurementDetail']['EndTime'])
            filter_dict = {}
            pixelsize_dict = {}
            if isinstance(doc['MeasurementDetail']['MeasurementChannel'], list):
                for channel_info in doc['MeasurementDetail']['MeasurementChannel']:
                    channel = int(channel_info['Ch'])
                    filter_pos = int(channel_info['FilterWheelPosition'])
                    filter_dict[channel] = filter_pos
                    pixelsize_dict[channel] = float(channel_info['HorizontalPixelDimension'])
            else:
                channel = int(doc['MeasurementDetail']['MeasurementChannel']['Ch'])
                filter_pos = int(doc['MeasurementDetail']['MeasurementChannel']['FilterWheelPosition'])
                filter_dict[channel] = filter_pos
                pixelsize_dict[channel] = float(doc['MeasurementDetail']['MeasurementChannel']['HorizontalPixelDimension'])
        for dict_ in list_record:
            type_ = dict_['Type']
            if type_ != 'ERR':
                timestamp = pd.to_datetime(dict_['Time'])
                column = int(dict_['Column'])
                row = int(dict_['Row'])
                timepoint = int(dict_['TimePoint'])
                field = int(dict_['FieldIndex'])
                z_idx = int(dict_['ZIndex'])
                timeline_idx = int(dict_['TimelineIndex'])
                action_idx = int(dict_['ActionIndex'])
                action = dict_['Action']
                channel = int(dict_['Ch'])
                color = filter_dict[channel]
                pixelsize = pixelsize_dict[channel]
                basename = dict_['#text']
                basename_split = basename.split('_')
                well = basename_split[-2]
                filename = dirname + basename
                if 'Fixed' in filename or 'fixed' in filename:
                    fixed = True
                else:
                    fixed = False
                condition = data_dict[well.replace('0', '')] if well.replace('0', '') in data_dict.keys() else None
                meta_imgs.append([filename, Title, condition, timestamp, BeginTime, EndTime, column, row,
                                  well, timepoint, field, z_idx, timeline_idx, action_idx, action, channel, color,
                                  fixed, pixelsize])

    meta_imgs = np.asarray(meta_imgs)
    columns = ['filename', 'title', 'plate', 'diff', 'condition', 'timestamp', 'begin', 'end', 'column', 'row', 'well',
               'timepoint', 'field', 'z_idx', 'timeline_idx', 'action_idx', 'action', 'channel', 'color',
               'fixed', 'pixelsize']
    df_imgs = pd.DataFrame(meta_imgs, columns=columns)

    # data summary
    plates, wells, fields = data_summary(df_imgs)

    return df_imgs, plates, wells, fields


def data_summary(df_imgs, print_summary=True):
    """
    Print a summary of unique values in the DataFrame.

    Parameters:
    - df_imgs (DataFrame): DataFrame containing image data

    Returns:
    None
    """

    plates, wells, fields = np.unique(df_imgs['plate']), np.unique(df_imgs['well']), np.unique(df_imgs['field'])
    timepoints, slices, channels = (np.unique(df_imgs['timepoint']), np.unique(df_imgs['z_idx']),
                                    np.unique(df_imgs['channel']))
    n_plates, n_wells, n_fields, n_timepoints, n_slices, n_channels = (len(plates), len(wells), len(fields),
                                                                       len(timepoints), len(slices), len(channels))

    if print_summary:
        print(f'{n_plates} plates, {n_wells} wells, {n_fields} fields, {n_timepoints} timepoints, {n_slices} slices, '
              f'{n_channels} channels')

    return plates, wells, fields


def filter_df(df_imgs, **kwargs):
    """
    Filter a DataFrame based on the provided conditions.

    Parameters:
        df_imgs (pandas.DataFrame): The DataFrame to be filtered.
        **kwargs: Arbitrary keyword arguments representing the filtering conditions.
                  The keys should correspond to column names in the DataFrame,
                  and the values can be either strings, integers, or datetime64.

    Returns:
        pandas.DataFrame: The filtered DataFrame.

    Example:
        df_filtered = get_data(df_imgs, well='A1', action='3D', channel=2, fixed=False)
    """
    query_parts = []
    for key, value in kwargs.items():
        if isinstance(value, str):
            query_parts.append(f"{key} == '{value}'")
        elif isinstance(value, pd.Timestamp):
            query_parts.append(f"{key} == '{value}'")
        else:
            query_parts.append(f"{key} == {value}")

    query_string = " and ".join(query_parts)
    df_filt = df_imgs.query(query_string).sort_values('timestamp', ignore_index=True)

    return df_filt


def compile_field(df_imgs, plate, well, field, folder_export, proj_mode='map'):
    """
    Compile data for a specific plate, well, and field from a DataFrame of images.

    Parameters:
    - df_imgs (DataFrame): DataFrame containing image metadata
    - plate (int): Plate number
    - well (str): Well identifier
    - field (int): Field number
    - folder_export (str): Folder path to export the compiled data
    - proj_mode (str, optional): Projection mode ('mip' or 'maxz'). Default is 'map'.

    Returns:
    None

    Raises:
    ValueError: If compiling of data fails

    Load metadata by df_meta = pd.read_csv(csv_file, parse_dates=['timestamps'], index_col=0)
    """
    df_i = filter_df(df_imgs, plate=plate, well=well, field=field)
    actions = df_i['action'].unique()
    begin_timestamps_i = np.sort(df_i['begin'].unique())
    for action in actions:  # actions (e.g. "2D" or "3D")
        df_i = filter_df(df_i, action=action)
        channels_i = df_i['channel'].unique()
        for channel_ij in channels_i:  # fluorescent channels
            df_ij = filter_df(df_i, channel=channel_ij, action=action)
            color_ij = df_ij.color.unique()[0]
            name_ij = f'plate{plate}_well{well}_field{field}_channel{channel_ij}_color{color_ij}_action{action}'
            if len(df_ij) > 0:
                data_ij = []
                conditions_ij = []
                timestamps_ij = []
                for begin in begin_timestamps_i:  # begin timepoints of measurements
                    df_ijk = filter_df(df_ij, begin=begin)
                    timepoints_ijk = np.sort(df_ijk['timepoint'].unique())
                    for tp in timepoints_ijk:  # timepoints per measurement
                        data_ijk = []
                        timestamps_ijk = []
                        conditions_ijk = []
                        slices_ijk = df_ijk['z_idx'].unique()
                        for z in slices_ijk:
                            df_ijkl = filter_df(df_ijk, timepoint=tp, z_idx=z)
                            if len(df_ijkl) != 1:
                                raise ValueError(f'Compiling of data failed. Len={len(df_ijkl)}.')
                            file_ijkl = df_ijkl['filename'].values[0]
                            data_ijkl = tifffile.imread(file_ijkl)
                            data_ijk.append(data_ijkl)
                            timestamps_ijk.append(df_ijkl['timestamp'])
                            conditions_ijk.append(df_ijkl['condition'])
                        timestamps_ij.append(np.min(timestamps_ijk))
                        conditions_ij.append(np.unique(conditions_ijk)[0])
                        data_ijk = np.asarray(data_ijk)
                        if proj_mode == 'mip':
                            data_ij.append(np.max(data_ijk, axis=0))
                        elif proj_mode == 'maxz':
                            data_ij.append(data_ijk[np.argmax(np.mean(data_ijk, axis=(1, 2)))])
                data_ij = np.asarray(data_ij)
                pixelsize = df_ij['pixelsize'][0]
                # save tif
                tifffile.imwrite(folder_export + name_ij + '.tif', data_ij, imagej=True,
                                 resolution=(1 / pixelsize, 1 / pixelsize),
                                 metadata={'unit': 'um', 'axes': 'ZYX', 'timestamps': timestamps_ij,
                                           'conditions': conditions_ij})


def get_well_name(well):
    col, row = re.findall(r'(\d+)_(\d+)', well)[0]
    number = str(col)
    letter = string.ascii_uppercase[int(row) - 1]
    return letter + number
