import re
import string

import pandas as pd


def filter_df(df_imgs, **kwargs):
    """Filter a DataFrame based on the provided conditions.

    Parameters:
        df_imgs (pandas.DataFrame): The DataFrame to be filtered.
        **kwargs: Arbitrary keyword arguments representing the filtering conditions.
                  The keys should correspond to column names in the DataFrame,
                  and the values can be either strings, integers, or datetime64.

    Returns:
        pandas.DataFrame: The filtered DataFrame.

    Example:
        df_filtered = filter_df(df_imgs, well='A1', action='3D', channel=2, fixed=False)
    """
    mask = pd.Series(True, index=df_imgs.index)
    for key, value in kwargs.items():
        mask &= df_imgs[key] == value

    df_filt = df_imgs[mask].sort_values('timestamp', ignore_index=True)
    return df_filt


def get_well_name(well):
    """Convert a numeric well identifier (e.g. '03_02') to alphanumeric format (e.g. 'B3').

    Parameters:
        well (str): Well identifier in 'col_row' format.

    Returns:
        str: Well name in 'LetterNumber' format (e.g. 'A1', 'B3').
    """
    col, row = re.findall(r'(\d+)_(\d+)', well)[0]
    number = str(int(col))
    letter = string.ascii_uppercase[int(row) - 1]
    return letter + number
