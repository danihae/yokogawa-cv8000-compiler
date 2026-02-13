import os

import numpy as np
import pandas as pd
import xmltodict


def parse_measurement_data(mlf_files):
    """Create a DataFrame with file metadata from Yokogawa CV8000 measurement files.

    Parameters:
        mlf_files (list[str]): Paths to MeasurementData.mlf files.

    Returns:
        tuple: (df_imgs, plates, wells, fields)
            - df_imgs (DataFrame): DataFrame containing image metadata.
            - plates (ndarray): Unique plate identifiers.
            - wells (ndarray): Unique well identifiers.
            - fields (ndarray): Unique field identifiers.
    """
    meta_imgs = []

    for mlf_file in mlf_files:
        # read MeasurementData.mlf file with single tif-file information
        with open(mlf_file) as fd:
            doc = xmltodict.parse(fd.read(),
                                  process_namespaces=True,
                                  namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},  # type: ignore[arg-type]
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
                    if line.strip():
                        key, value = line.split(' - ')
                        data_dict[key] = value
        else:
            data_dict = {}
        # read MeasurementDetail.mrf file with metadata
        mrf_file = dirname + 'MeasurementDetail.mrf'
        with open(mrf_file, encoding="utf-8-sig") as fd:
            xml_data = fd.read()
            doc = xmltodict.parse(xml_data,
                                  process_namespaces=True,
                                  namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},  # type: ignore[arg-type]
                                  attr_prefix="", cdata_key="Value"
                                  )
            title = doc['MeasurementDetail']['Title']
            begin_time = pd.to_datetime(doc['MeasurementDetail']['BeginTime'])
            end_time = pd.to_datetime(doc['MeasurementDetail']['EndTime'])
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
                basename = dict_['Value']
                basename_split = basename.split('_')
                well = basename_split[-2]
                filename = dirname + basename
                fixed = 'fixed' in filename.lower()
                condition = data_dict.get(well.replace('0', ''))
                diff = timestamp - begin_time
                meta_imgs.append([filename, title, title, diff, condition, timestamp,
                                  begin_time, end_time, column, row, well, timepoint,
                                  field, z_idx, timeline_idx, action_idx, action,
                                  channel, color, fixed, pixelsize])

    columns = ['filename', 'title', 'plate', 'diff', 'condition', 'timestamp',
               'begin', 'end', 'column', 'row', 'well', 'timepoint', 'field',
               'z_idx', 'timeline_idx', 'action_idx', 'action', 'channel',
               'color', 'fixed', 'pixelsize']
    df_imgs = pd.DataFrame(meta_imgs, columns=columns)

    # data summary
    plates, wells, fields = data_summary(df_imgs)

    return df_imgs, plates, wells, fields


def data_summary(df_imgs, print_summary=True):
    """Print a summary of unique values in the DataFrame.

    Parameters:
        df_imgs (DataFrame): DataFrame containing image data.
        print_summary (bool): Whether to print the summary. Default is True.

    Returns:
        tuple: (plates, wells, fields) arrays of unique values.
    """
    plates = df_imgs['plate'].unique()
    wells = df_imgs['well'].unique()
    fields = df_imgs['field'].unique()
    timepoints = df_imgs['timepoint'].unique()
    slices = df_imgs['z_idx'].unique()
    channels = df_imgs['channel'].unique()

    if print_summary:
        print(f'{len(plates)} plates, {len(wells)} wells, {len(fields)} fields, '
              f'{len(timepoints)} timepoints, {len(slices)} slices, {len(channels)} channels')

    return plates, wells, fields
