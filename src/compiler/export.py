import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List
from typing import Literal, Tuple, Union

import dask
import dask.array as da
import numpy as np
import pandas as pd
import tifffile
import xarray as xr
from scipy import ndimage as ndi
from tqdm import tqdm

from .metadata import CellVoyagerAcquisition
from .discovery import logger


def _get_well_id(row: int, column: int) -> str:
    """
    Converts row and column indices to a standard well ID format.

    Parameters
    ----------
    row : int
        Row index (0-based).
    column : int
        Column index (0-based).

    Returns
    -------
    str
        Well ID in format like "A01" or "B12".

    Examples
    --------
    >>> _get_well_id(0, 0)
    'A01'
    >>> _get_well_id(1, 11)
    'B12'
    """
    return f"{chr(65 + row)}{column + 1:02d}"


def _all_equal(iterator):
    """
    Checks if all elements in an iterator are equal.

    Parameters
    ----------
    iterator : iterable
        The iterable to check.

    Returns
    -------
    bool
        True if all elements are equal, False otherwise. Returns True for empty iterators.
    """
    iterator = iter(iterator)
    try:
        first = next(iterator)
    except StopIteration:
        return True
    return all(np.array_equal(first, x) if isinstance(first, np.ndarray) else first == x for x in iterator)


def _read_tif(path: Union[str, Path]):
    """
    Reads a TIFF image from a file path.

    Parameters
    ----------
    path : str or Path
        Path to the TIFF file.

    Returns
    -------
    ndarray
        NumPy array containing the image data.
    """
    return tifffile.imread(str(Path(path)))

def _tenengrad_focus(stack: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    """Per-slice Tenengrad focus measure: smoothed squared in-plane Sobel gradient.

    Sobel and the optional Gaussian are applied independently to each 2D (Y, X)
    slice — leading dims (e.g. T, C, Z) are iterated over flat. Treating the
    operators as strictly 2D matters: applying ``ndi.sobel`` directly to a
    higher-dim array would also smooth along non-spatial axes.

    Parameters
    ----------
    stack : np.ndarray
        Input array with shape (..., Y, X).
    sigma : float, optional
        Gaussian smoothing applied to the per-pixel focus measure.
    """
    s = stack.astype(np.float32, copy=False)
    flat = s.reshape(-1, s.shape[-2], s.shape[-1])
    out = np.empty_like(flat)
    for i in range(flat.shape[0]):
        gx = ndi.sobel(flat[i], axis=0)
        gy = ndi.sobel(flat[i], axis=1)
        m = gx * gx + gy * gy
        if sigma > 0:
            m = ndi.gaussian_filter(m, sigma=sigma)
        out[i] = m
    return out.reshape(s.shape)


def osbm_projection(arr: xr.DataArray, sigma: float = 2.0) -> xr.DataArray:
    """
    Out-of-focus-Suppressed Brightfield Maximum (OSBM) projection.

    For each (Y, X) pixel, computes a per-Z focus measure (Tenengrad: smoothed
    squared in-plane Sobel gradient magnitude), normalizes it across Z to
    obtain weights, and returns the weighted sum across Z. Pixels in
    well-focused slices dominate while out-of-focus content is suppressed.

    Parameters
    ----------
    arr : xr.DataArray
        DataArray with a 'Z' dimension and spatial 'Y', 'X' dimensions.
    sigma : float, optional
        Gaussian sigma used to smooth the focus measure (default 2.0).

    Returns
    -------
    xr.DataArray
        DataArray with the 'Z' dimension removed, same dtype as input.
    """
    in_dtype = arr.dtype
    z_axis = arr.get_axis_num('Z')

    s = arr.values.astype(np.float32, copy=False)
    s_zlast = np.moveaxis(s, z_axis, -3)  # (..., Z, Y, X)

    focus = _tenengrad_focus(s_zlast, sigma=sigma)
    weights = focus / (focus.sum(axis=-3, keepdims=True) + 1e-9)
    proj = (s_zlast * weights).sum(axis=-3)  # (..., Y, X)

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        proj = np.clip(proj, info.min, info.max).astype(in_dtype)
    else:
        proj = proj.astype(in_dtype)

    output_dims = [dim for dim in arr.dims if dim != 'Z']
    output_coords = {dim: arr.coords[dim] for dim in output_dims if dim in arr.coords}
    return xr.DataArray(proj, coords=output_coords, dims=output_dims)

def _entropy_projection(arr: xr.DataArray, mode: str) -> xr.DataArray:
    """Helper function for both max and min entropy projections."""
    def entropy(image_slice: xr.DataArray) -> float:
        """Calculate the Shannon entropy of a single 2D image slice."""
        # Use .values to get the NumPy array for numpy.histogram
        hist, _ = np.histogram(image_slice.values.ravel(), bins=256, density=True)
        hist = hist[hist > 0]
        return -np.sum(hist * np.log2(hist))

    # Get dimension sizes from the xarray object
    T, C, Z = arr.sizes['T'], arr.sizes['C'], arr.sizes['Z']

    # Pre-allocate a NumPy array to store the result
    projected_data = np.empty((T, C, arr.sizes['Y'], arr.sizes['X']), dtype=arr.dtype)

    for t in range(T):
        for c in range(C):
            # Calculate entropy for each Z-slice
            entropies = [entropy(arr.isel(T=t, C=c, Z=z)) for z in range(Z)]

            # Find the best Z-slice based on the desired mode
            if mode == 'max':
                best_z = np.argmax(entropies)
            else: # mode == 'min'
                best_z = np.argmin(entropies)

            projected_data[t, c] = arr.isel(T=t, C=c, Z=best_z).values

    # Construct and return a new xarray.DataArray with the correct dimensions and coordinates
    output_coords = {dim: arr.coords[dim] for dim in arr.dims if dim != 'Z'}
    return xr.DataArray(
        projected_data,
        coords=output_coords,
        dims=list(output_coords.keys())
    )

def max_entropy_projection(arr: xr.DataArray) -> xr.DataArray:
    """
    Performs a Maximum Entropy Projection. Returns an xarray.DataArray.
    """
    return _entropy_projection(arr, mode='max')

def min_entropy_projection(arr: xr.DataArray) -> xr.DataArray:
    """
    Performs a Minimum Entropy Projection. Returns an xarray.DataArray.
    """
    return _entropy_projection(arr, mode='min')


def _make_correction_func(dark: np.ndarray, gain: np.ndarray):
    """Create an image correction function with bound dark/gain arrays."""
    def _correct(img_u16):
        img = img_u16.astype(np.float32)
        img_corr = (img - dark) * gain
        return np.clip(img_corr, 0, 65535).astype(np.uint16)
    return _correct


def write_fieldstack(
        key: Tuple[int, int, int, int, int, str],
        df: pd.DataFrame,
        acquisition_metadata: Union[CellVoyagerAcquisition, list[CellVoyagerAcquisition]],
        out_dir: Union[Path, str],
        title: str | None = None,
        *,
        z_mode: Literal["keep", "mip", "maxz", "osbm", "max_entropy", "min_entropy"] = "keep",
        z_mode_BF: Literal["keep", "osbm"] = "keep",
        correct: bool = True,
        compress: str | None = "lzw",
        overwrite: bool = False,
        dry_run: bool = False,
) -> None:
    """
    Writes a 5D image stack for a single field as a well-annotated OME-TIFF file.

    Processes microscopy image data from a DataFrame, applies optional corrections
    and Z-projections lazily using xarray, and writes the result using TiffFile.

    Parameters
    ----------
    key : tuple of (int, int, int, int, int, str)
        Tuple identifying the data: (row, column, field_index, timeline_index, action_index, action_name).
    df : DataFrame
        Pandas DataFrame with image records. Expected columns: 'ch', 'time_point', 'z_index', 'tif_path',
        'timestamp', 'acquisition_index', 'begin_time'.
    acquisition_metadata : CellVoyagerAcquisition or list of CellVoyagerAcquisition
        Metadata for the experiment acquisitions.
    out_dir : Path or str
        Destination folder for the output file.
    title : str or None, optional
        Optional prefix for the output filename. If None (default), no prefix is added.
    z_mode : {"keep", "mip", "maxz", "osbm", "max_entropy", "min_entropy"}, optional
        Z-projection method for fluorescence channels (default is "keep").
    z_mode_BF : {"keep", "osbm"}, optional
        Z-projection method for brightfield channels (default is "keep").
    correct : bool, optional
        If True, apply dark and flat-field corrections (default is True).
    compress : str or None, optional
        Compression algorithm for the TIFF file (default is "lzw").
    overwrite : bool, optional
        If True, overwrite existing files (default is False).
    dry_run : bool, optional
        If True, skip file I/O but perform other steps (default is False).

    Returns
    -------
    None

    Raises
    ------
    ValueError
        If acquisitions have inconsistent channels or Z-slices, or pixel sizes differ.
    AssertionError
        If DataFrame size does not match expected image count.
    """
    row, column, field_index, timeline_index, action_index, action = key
    well_id = _get_well_id(row, column)

    current_z_mode = z_mode if action != "BF3D" else z_mode_BF

    begin_times, acquisition_indices = (
        df[["begin_time", "acquisition_index"]].drop_duplicates().sort_values("begin_time").T.values
    )

    _channels_per_acq = [df[df["begin_time"] == bt]["ch"].unique() for bt in begin_times]
    _z_indices_per_acq = [len(df[df["begin_time"] == bt]["z_index"].unique()) for bt in begin_times]

    if not _all_equal(_channels_per_acq):
        raise ValueError("Acquisitions have different channels and cannot be merged.")
    if not _all_equal(_z_indices_per_acq):
        raise ValueError("Acquisitions have an unequal number of Z-slices and cannot be merged.")

    channels_list = _channels_per_acq[0]
    n_channels = len(channels_list)
    n_z_indices = _z_indices_per_acq[0]
    n_time_points = sum(len(df[df["begin_time"] == bt]["time_point"].unique()) for bt in begin_times)

    expected_images = n_time_points * n_channels * n_z_indices
    if len(df) != expected_images:
        error_msg = (f"DataFrame size mismatch: got {len(df)} records, "
                     f"but expected {n_time_points} timepoints × {n_channels} channels × "
                     f"{n_z_indices} Z-indices = {expected_images}")
        raise AssertionError(error_msg)

    # Get image dimensions lazily from first image
    first_img_path = df.iloc[0]['tif_path']
    sample_img = _read_tif(first_img_path)  # Load only one for shape
    n_y, n_x = sample_img.shape

    # Prepare coordinates for xarray
    time_coords = np.arange(n_time_points)
    channel_coords = channels_list
    z_coords = np.arange(n_z_indices)
    y_coords = np.arange(n_y)
    x_coords = np.arange(n_x)

    # Create a list of lazy loaders for each image
    def load_and_correct(tif_path, correct_func=None):
        img = _read_tif(tif_path)
        if correct_func:
            img = correct_func(img)
        return img

    # Pre-compute correction functions once per (acquisition, channel).
    # Without this, each (timepoint, channel) iteration would re-read dark/flat
    # TIFFs and recompute the gain, multiplying disk I/O by n_timepoints.
    correction_funcs: dict[Tuple[int, int], object] = {}
    if correct and "BF" not in action:
        for begin_time, acquisition_index in zip(begin_times, acquisition_indices):
            meta = acquisition_metadata[acquisition_index]
            df_acq = df[df["begin_time"] == begin_time]
            parent = os.path.dirname(str(df_acq.iloc[0]["tif_path"]))
            for ch in np.sort(df_acq["ch"].unique()):
                key = (int(acquisition_index), int(ch))
                if key in correction_funcs:
                    continue
                camera = meta.measurement_detail.measurement_channel[ch - 1].camera_number
                dark_path = parent + "/" + f"DC_DCAM#{camera}_CAM{camera}.tif"
                dark = _read_tif(dark_path).astype(np.float32)
                flat_path = parent + "/" + meta.measurement_detail.measurement_channel[
                    ch - 1].shading_correction_source
                flat = _read_tif(flat_path).astype(np.float32)
                ff = flat - dark
                gain = np.mean(ff) / ff
                gain[np.isinf(gain)] = 0
                correction_funcs[key] = _make_correction_func(dark, gain)

    # Build xarray with lazy loading using Dask delays
    data = []
    timestamps = []
    time_point_counter = 0
    for begin_time, acquisition_index in zip(begin_times, acquisition_indices):
        df_acq = df[df["begin_time"] == begin_time]
        time_points = np.sort(df_acq["time_point"].unique())
        channels = np.sort(df_acq["ch"].unique())
        z_indices = np.sort(df_acq["z_index"].unique())

        for t, time_point in enumerate(time_points):
            min_time = None
            t_data = []
            for c, ch in enumerate(channels):
                _correct_img = correction_funcs.get((int(acquisition_index), int(ch)))

                c_data = []
                for z, z_index in enumerate(z_indices):
                    df_img = df_acq[
                        (df_acq["time_point"] == time_point) &
                        (df_acq["ch"] == ch) &
                        (df_acq["z_index"] == z_index)
                        ]
                    if len(df_img) != 1:
                        logger.warning(
                            f"Expected 1 file for (time={time_point}, ch={ch}, z={z_index}), found {len(df_img)}")
                        continue

                    tif_path_full = df_img.iloc[0]["tif_path"]
                    time_current = df_img.iloc[0]["time"]
                    if min_time is None:
                        min_time = time_current
                    elif time_current < min_time:
                        min_time = time_current

                    delayed_img = da.from_delayed(dask.delayed(load_and_correct)(tif_path_full, _correct_img),
                                                  shape=(n_y, n_x), dtype=np.uint16)
                    c_data.append(delayed_img)

                t_data.append(da.stack(c_data, axis=0))  # Stack Z

            data.append(da.stack(t_data, axis=0))  # Stack C
            timestamps.append(min_time)

        time_point_counter += len(time_points)

    # Create lazy xarray DataArray
    arr = xr.DataArray(da.stack(data, axis=0), dims=['T', 'C', 'Z', 'Y', 'X'],
                       coords={'T': time_coords[:len(data)], 'C': channel_coords, 'Z': z_coords, 'Y': y_coords,
                               'X': x_coords})

    # Apply Z-projection lazily
    if current_z_mode == "mip":
        arr = arr.max(dim='Z')
    elif current_z_mode == "maxz":
        mean_intensity = arr.mean(dim=['T', 'C', 'Y', 'X'])
        best_z_lazy = mean_intensity.argmax(dim='Z')
        best_z_concrete = best_z_lazy.compute()
        arr = arr.isel(Z=best_z_concrete)
    elif current_z_mode in ["osbm", "max_entropy", "min_entropy"]:
        # Projection functions need the full Z stack per call, so collapse the
        # Z axis into a single dask chunk before mapping.
        arr = arr.chunk({'Z': -1})

        # This template correctly defines the expected output shape and dimensions
        template = arr.isel(Z=0, drop=True)

        # Select the appropriate projection function
        if current_z_mode == "osbm":
            projection_func = osbm_projection
        elif current_z_mode == "max_entropy":
            projection_func = max_entropy_projection
        elif current_z_mode == "min_entropy":
            projection_func = min_entropy_projection

        arr = arr.map_blocks(projection_func, template=template)

    # Squeeze dimensions
    arr = arr.squeeze()
    axes = "".join(arr.dims)

    # Metadata handling with non-standard keys merged into main dict
    if isinstance(acquisition_metadata, list):
        acquisition_metadata = acquisition_metadata[acquisition_indices[0]]

    channel_list = acquisition_metadata.measurement_setting.channel_list.channel
    pixel_sizes_x = [ch.horizontal_pixel_dimension for ch in
                     acquisition_metadata.measurement_detail.measurement_channel]
    pixel_sizes_y = [ch.vertical_pixel_dimension for ch in acquisition_metadata.measurement_detail.measurement_channel]

    if len(set(pixel_sizes_y)) > 1 or len(set(pixel_sizes_x)) > 1:
        raise ValueError("Pixel sizes differ between channels and cannot be reconciled.")

    physical_size_y = pixel_sizes_y[0] if pixel_sizes_y else 1.0
    physical_size_x = pixel_sizes_x[0] if pixel_sizes_x else 1.0

    # Combined metadata (standard OME + non-standard)
    ome_metadata = {
        "axes": axes,
        "PhysicalSizeY": physical_size_y,
        "PhysicalSizeX": physical_size_x,
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeXUnit": "µm",
        "Channels": [],
        "PlateType": acquisition_metadata.measurement_detail.measurement_sample_plate.name,
        "Operator": acquisition_metadata.measurement_detail.operator_name,
        "Timestamps": timestamps,
        "WellID": well_id,
        "FieldIndex": field_index,
        "ActionIndex": action_index,
        "Action": action,
        "ZMode": current_z_mode
    }

    for i, ch_num in enumerate(channels_list):
        if i < len(channel_list):
            ch_meta = channel_list[i]
            ch_dict = {
                "Name": f"Channel_{ch_num}",
                "Magnification": ch_meta.magnification,
                "Objective": ch_meta.objective,
                "ExposureTime": ch_meta.exposure_time,
                "Acquisition": ch_meta.acquisition,
                "Method": ch_meta.method,
                "LightSource": ch_meta.light_source_name,
                "Fluorophore": ch_meta.fluorophore
            }
        else:
            ch_dict = {"Name": f"Channel_{ch_num}"}
        ome_metadata["Channels"].append(ch_dict)

    output_directory = Path(out_dir)
    output_directory.mkdir(exist_ok=True)
    magnification = ome_metadata["Channels"][0].get('Magnification', 0)
    prefix = f"{title}_" if title else ""
    fname = f"{prefix}{well_id}_F{field_index:02d}_L{timeline_index}_A{action_index}_{action}_{magnification}x.ome.tif"
    destination = output_directory / fname

    if destination.exists() and not overwrite:
        logger.warning("Skipping existing file: %s", destination)
        return

    logger.info("Writing %s (axes=%s, shape=%s)", destination, axes, arr.shape)

    def clean_metadata_for_json(obj):
        """
        Recursively converts NumPy types to JSON-serializable Python types.

        Parameters
        ----------
        obj : any
            Object to clean (dict, list, NumPy array, etc.).

        Returns
        -------
        any
            Cleaned object suitable for JSON serialization.
        """
        if isinstance(obj, dict):
            return {k: clean_metadata_for_json(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean_metadata_for_json(item) for item in obj]
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    clean_metadata = clean_metadata_for_json(ome_metadata)

    # Compute and write using tifffile
    if not dry_run:
        arr_computed = arr.compute()  # Compute lazy data
        tifffile.imwrite(
            destination,
            arr_computed.values,  # Extract NumPy array from xarray
            metadata=clean_metadata,
            bigtiff=True,
            compression=compress,
        )


def process_fieldstacks_parallel(
    fieldstacks: pd.core.groupby.generic.DataFrameGroupBy,
    acquisitions: List['CellVoyagerAcquisition'],
    out_dir: Union[str, Path],
    *,
    title: str | None = None,
    z_mode: str = "maxz",
    z_mode_BF: str = "keep",
    correct: bool = True,
    overwrite: bool = True,
    max_workers: int = 4,
) -> None:
    """
    Processes and writes field stacks to OME-TIFF files in parallel.

    This function wraps the write_fieldstack call and uses a ThreadPoolExecutor
    to parallelize the processing of each independent field stack.

    Parameters
    ----------
    fieldstacks : pandas.core.groupby.generic.DataFrameGroupBy
        A DataFrame grouped by field stack identifiers.
    acquisitions : List[CellVoyagerAcquisition]
        The list of acquisition metadata objects.
    out_dir : Path
        The destination directory for the output OME-TIFF files.
    title : str or None, optional
        Optional prefix for the output filenames. If None (default), no prefix is added.
    z_mode : str, optional
        The Z-projection mode for fluorescence channels, by default "maxz".
    z_mode_BF : str, optional
        The Z-projection mode for brightfield channels, by default "keep".
    correct : bool, optional
        Whether to apply image corrections, by default True.
    overwrite : bool, optional
        Whether to overwrite existing files, by default True.
    max_workers : int, optional
        The number of parallel threads to use, by default 4. Adjust this
        based on your system's core count and I/O capacity.
    """

    def _process_single_stack(args):
        """Helper function to unpack arguments for write_fieldstack."""
        key, stack_df = args
        write_fieldstack(
            key,
            stack_df,
            acquisitions,
            out_dir=out_dir,
            title=title,
            z_mode=z_mode,
            z_mode_BF=z_mode_BF,
            correct=correct,
            overwrite=overwrite,
        )

    # Use ThreadPoolExecutor to run the processing in parallel
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # The list() consumes the iterator, ensuring all tasks complete.
        # tqdm provides a progress bar over the submitted tasks.
        list(
            tqdm(
                executor.map(_process_single_stack, fieldstacks),
                total=len(fieldstacks),
                desc="Processing field stacks"
            )
        )
