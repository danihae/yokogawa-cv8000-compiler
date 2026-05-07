import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Optional
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


TileMode = Literal["per-field", "stitch"]


def _get_well_id(row: int, column: int) -> str:
    """
    Converts row and column indices to a standard well ID format.

    Parameters
    ----------
    row : int
        Row index (1-based, as stored in the Yokogawa MLF).
    column : int
        Column index (1-based, as stored in the Yokogawa MLF).

    Returns
    -------
    str
        Well ID in format like "A01" or "B12".

    Examples
    --------
    >>> _get_well_id(1, 1)
    'A01'
    >>> _get_well_id(2, 12)
    'B12'
    """
    return f"{chr(64 + row)}{column:02d}"


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


def _empirical_time_metadata(timestamps: list) -> dict:
    """Derive empirical per-frame intervals and framerate from absolute timestamps.

    Image-level ``Time`` strings live in the MLF; we keep the earliest one per
    timepoint as ``timestamps``. From those we compute:

    - ``RelativeTimes`` — seconds since the first timepoint, one per T.
    - ``FrameIntervals`` — successive deltas in seconds.
    - ``TimeIncrement`` — median frame interval (s); robust to a missed frame.
    - ``FramerateHz`` — ``1 / TimeIncrement``.

    Returns ``{}`` when fewer than two parseable timestamps are available.
    """
    parsed: list = []
    for ts in timestamps:
        if ts is None:
            parsed.append(None)
            continue
        try:
            parsed.append(pd.to_datetime(ts))
        except Exception:
            parsed.append(None)

    valid = [t for t in parsed if t is not None]
    if len(valid) < 2:
        return {}

    t0 = valid[0]
    relative = [
        float((t - t0).total_seconds()) if t is not None else None for t in parsed
    ]
    intervals = [
        float((parsed[i] - parsed[i - 1]).total_seconds())
        for i in range(1, len(parsed))
        if parsed[i] is not None and parsed[i - 1] is not None
    ]
    out: dict = {
        "RelativeTimes": relative,
        "RelativeTimesUnit": "s",
    }
    if intervals:
        out["FrameIntervals"] = intervals
        median_dt = float(np.median(intervals))
        out["TimeIncrement"] = median_dt
        out["TimeIncrementUnit"] = "s"
        if median_dt > 0:
            out["FramerateHz"] = 1.0 / median_dt
    return out


def _timeline_metadata(
    acquisition_metadata: List[CellVoyagerAcquisition],
    acquisition_indices,
    timeline_index: int,
) -> dict:
    """Pull configured timing values from the .mes ``Timeline`` element.

    Yokogawa's BTS XML defines ``Period``, ``Interval``, and ``ExpectedTime``
    per ``Timeline``; the units are not explicit in the schema but are
    surfaced as raw values for reference. When merging across acquisitions, we
    use the timeline from the first acquisition in the group.
    """
    if len(acquisition_indices) == 0:
        return {}
    acq = acquisition_metadata[int(acquisition_indices[0])]
    timelines = acq.measurement_setting.timelapse.timeline
    if timeline_index < 1 or timeline_index > len(timelines):
        return {}
    tl = timelines[timeline_index - 1]
    return {
        "TimelineName": tl.name,
        "TimelinePeriod": tl.period,
        "TimelineInterval": tl.interval,
        "TimelineExpectedTime": tl.expected_time,
    }


def _feather_mask(h: int, w: int, overlap_y: int, overlap_x: int) -> np.ndarray:
    """2D blending mask with linear ramps in the overlap regions at each tile edge.

    Always strictly positive, so overlap-aware sums can be normalised by
    accumulated weights without divide-by-zero handling.
    """
    ramp_y = np.ones(h, dtype=np.float32)
    if overlap_y > 0:
        ramp = (np.arange(overlap_y, dtype=np.float32) + 1.0) / (overlap_y + 1.0)
        ramp_y[:overlap_y] = ramp
        ramp_y[h - overlap_y:] = ramp[::-1]
    ramp_x = np.ones(w, dtype=np.float32)
    if overlap_x > 0:
        ramp = (np.arange(overlap_x, dtype=np.float32) + 1.0) / (overlap_x + 1.0)
        ramp_x[:overlap_x] = ramp
        ramp_x[w - overlap_x:] = ramp[::-1]
    return ramp_y[:, None] * ramp_x[None, :]


def _stitch_tile_arrays(
    tile_arrays: dict[Tuple[int, int], xr.DataArray],
    overlap_x: int,
    overlap_y: int,
) -> xr.DataArray:
    """Place a regular grid of tiles into a single mosaic with feathered blending.

    Tiles must share Y/X shape. (tx, ty) keys are 1-based tile indices on the
    stage grid; missing tiles in the rectangle simply remain at zero in the
    output.

    The output replaces the (Y, X) dims with mosaic dims of the same names.
    All other dims (T, C, Z, ...) are preserved.
    """
    keys = list(tile_arrays.keys())
    txs = sorted({k[0] for k in keys})
    tys = sorted({k[1] for k in keys})
    sample = next(iter(tile_arrays.values()))
    h = int(sample.sizes['Y'])
    w = int(sample.sizes['X'])
    step_x = max(w - overlap_x, 1)
    step_y = max(h - overlap_y, 1)
    out_h = (len(tys) - 1) * step_y + h
    out_w = (len(txs) - 1) * step_x + w

    other_dims = [d for d in sample.dims if d not in ('Y', 'X')]
    other_shape = tuple(int(sample.sizes[d]) for d in other_dims)

    accum = np.zeros(other_shape + (out_h, out_w), dtype=np.float32)
    weight_acc = np.zeros((out_h, out_w), dtype=np.float32)

    feather = _feather_mask(h, w, overlap_y, overlap_x)

    in_dtype = sample.dtype

    tx_pos = {tx: i for i, tx in enumerate(txs)}
    ty_pos = {ty: i for i, ty in enumerate(tys)}

    for (tx, ty), tile in tile_arrays.items():
        # Materialise into numpy with axes (other_dims..., Y, X)
        tile_np = tile.transpose(*other_dims, 'Y', 'X').values.astype(np.float32, copy=False)
        y0 = ty_pos[ty] * step_y
        x0 = tx_pos[tx] * step_x
        accum[..., y0:y0 + h, x0:x0 + w] += tile_np * feather
        weight_acc[y0:y0 + h, x0:x0 + w] += feather

    weight_acc = np.maximum(weight_acc, 1e-9)
    mosaic = accum / weight_acc

    if np.issubdtype(in_dtype, np.integer):
        info = np.iinfo(in_dtype)
        mosaic = np.clip(mosaic, info.min, info.max).astype(in_dtype)
    else:
        mosaic = mosaic.astype(in_dtype)

    coords = {d: sample.coords[d] for d in other_dims if d in sample.coords}
    coords['Y'] = np.arange(out_h)
    coords['X'] = np.arange(out_w)

    return xr.DataArray(mosaic, dims=other_dims + ['Y', 'X'], coords=coords)


def _build_correction_funcs(
    df: pd.DataFrame,
    acquisition_metadata: list[CellVoyagerAcquisition],
    begin_times,
    acquisition_indices,
    action: str,
) -> dict[Tuple[int, int], object]:
    """Return per-(acquisition, channel) shading-correction callables.

    Skips the correction (returning no entry) for any channel whose dark or
    flat-field reference TIFF is missing, rather than crashing the run.
    """
    correction_funcs: dict[Tuple[int, int], object] = {}
    if "BF" in action:
        return correction_funcs
    for begin_time, acquisition_index in zip(begin_times, acquisition_indices):
        meta = acquisition_metadata[int(acquisition_index)]
        df_acq = df[df["begin_time"] == begin_time]
        parent = os.path.dirname(str(df_acq.iloc[0]["tif_path"]))
        for ch in np.sort(df_acq["ch"].unique()):
            key = (int(acquisition_index), int(ch))
            if key in correction_funcs:
                continue
            try:
                ch_meta = meta.measurement_detail.measurement_channel[int(ch) - 1]
                camera = ch_meta.camera_number
                shading = ch_meta.shading_correction_source
                if not shading:
                    logger.warning(
                        "No shading correction source for ch=%s in acquisition %s; skipping correction",
                        ch, acquisition_index,
                    )
                    continue
                dark_path = os.path.join(parent, f"DC_DCAM#{camera}_CAM{camera}.tif")
                flat_path = os.path.join(parent, shading)
                if not os.path.exists(dark_path) or not os.path.exists(flat_path):
                    logger.warning(
                        "Missing correction TIFF for ch=%s (dark=%s flat=%s); skipping correction",
                        ch, dark_path, flat_path,
                    )
                    continue
                dark = _read_tif(dark_path).astype(np.float32)
                flat = _read_tif(flat_path).astype(np.float32)
                ff = flat - dark
                gain = np.mean(ff) / ff
                gain[np.isinf(gain)] = 0
                correction_funcs[key] = _make_correction_func(dark, gain)
            except Exception as exc:
                logger.warning(
                    "Failed to build correction for ch=%s in acquisition %s: %s",
                    ch, acquisition_index, exc,
                )
    return correction_funcs


def _build_tile_array(
    df: pd.DataFrame,
    correction_funcs: dict[Tuple[int, int], object],
    begin_times,
    acquisition_indices,
    n_y: int,
    n_x: int,
) -> Tuple[xr.DataArray, list]:
    """Assemble a lazy (T, C, Z, Y, X) DataArray from a DataFrame for one tile/field.

    Returns the DataArray plus a list of per-timepoint timestamps (the earliest
    image time inside each timepoint group).
    """
    def load_and_correct(tif_path, correct_func=None):
        img = _read_tif(tif_path)
        if correct_func is not None:
            img = correct_func(img)
        return img

    data = []
    timestamps: list = []

    channels_list = df["ch"].unique()
    channels_list.sort()
    n_z_indices = len(df["z_index"].unique())

    for begin_time, acquisition_index in zip(begin_times, acquisition_indices):
        df_acq = df[df["begin_time"] == begin_time]
        if df_acq.empty:
            continue
        time_points = np.sort(df_acq["time_point"].unique())
        z_indices = np.sort(df_acq["z_index"].unique())

        for time_point in time_points:
            min_time = None
            t_data = []
            for ch in channels_list:
                _correct_img = correction_funcs.get((int(acquisition_index), int(ch)))

                c_data = []
                for z_index in z_indices:
                    df_img = df_acq[
                        (df_acq["time_point"] == time_point)
                        & (df_acq["ch"] == ch)
                        & (df_acq["z_index"] == z_index)
                    ]
                    if len(df_img) != 1:
                        logger.warning(
                            "Expected 1 file for (time=%s, ch=%s, z=%s), found %s",
                            time_point, ch, z_index, len(df_img),
                        )
                        continue

                    tif_path_full = df_img.iloc[0]["tif_path"]
                    time_current = df_img.iloc[0]["time"]
                    if min_time is None or time_current < min_time:
                        min_time = time_current

                    delayed_img = da.from_delayed(
                        dask.delayed(load_and_correct)(tif_path_full, _correct_img),
                        shape=(n_y, n_x),
                        dtype=np.uint16,
                    )
                    c_data.append(delayed_img)

                t_data.append(da.stack(c_data, axis=0))  # Z

            data.append(da.stack(t_data, axis=0))  # C
            timestamps.append(min_time)

    n_time_points = len(data)
    arr = xr.DataArray(
        da.stack(data, axis=0),
        dims=['T', 'C', 'Z', 'Y', 'X'],
        coords={
            'T': np.arange(n_time_points),
            'C': channels_list,
            'Z': np.arange(n_z_indices),
            'Y': np.arange(n_y),
            'X': np.arange(n_x),
        },
    )
    return arr, timestamps


def _apply_z_projection(arr: xr.DataArray, current_z_mode: str) -> xr.DataArray:
    """Apply a Z-projection lazily according to ``current_z_mode``."""
    if current_z_mode == "mip":
        return arr.max(dim='Z')
    if current_z_mode == "maxz":
        mean_intensity = arr.mean(dim=['T', 'C', 'Y', 'X'])
        best_z_concrete = mean_intensity.argmax(dim='Z').compute()
        return arr.isel(Z=best_z_concrete)
    if current_z_mode in ("osbm", "max_entropy", "min_entropy"):
        arr = arr.chunk({'Z': -1})
        template = arr.isel(Z=0, drop=True)
        if current_z_mode == "osbm":
            projection_func = osbm_projection
        elif current_z_mode == "max_entropy":
            projection_func = max_entropy_projection
        else:
            projection_func = min_entropy_projection
        return arr.map_blocks(projection_func, template=template)
    return arr


def write_fieldstack(
        key: Tuple,
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
        tile_mode: TileMode = "per-field",
) -> None:
    """
    Writes a 5D image stack for a single field — or a stitched mosaic of tiles —
    as a well-annotated OME-TIFF file.

    Parameters
    ----------
    key : tuple
        Group identifier. For ``tile_mode="per-field"``: ``(row, column,
        field_index, timeline_index, action_index, action)``. For
        ``tile_mode="stitch"``: ``(row, column, partial_tile_index,
        timeline_index, action_index, action)``.
    df : DataFrame
        Pandas DataFrame with image records. Expected columns: 'ch', 'time_point', 'z_index', 'tif_path',
        'time', 'acquisition_index', 'begin_time'. For tiled stitching also requires
        'tile_x_index', 'tile_y_index', 'partial_tile_index'.
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
    tile_mode : {"per-field", "stitch"}, optional
        How to handle tiled acquisitions. ``"per-field"`` writes one file per
        field. ``"stitch"`` blends a tile grid (TileXIndex × TileYIndex) into a
        single mosaic per partial-tile group.

    Returns
    -------
    None
    """
    if len(key) == 7:
        row, column, group_id, timeline_index, action_index, action, split_acq_idx = key
    else:
        row, column, group_id, timeline_index, action_index, action = key
        split_acq_idx = None
    well_id = _get_well_id(row, column)

    current_z_mode = z_mode if action != "BF3D" else z_mode_BF

    # Determine tiling: stitch only when requested AND records carry tile indices.
    is_tiled_group = (
        tile_mode == "stitch"
        and "tile_x_index" in df.columns
        and df["tile_x_index"].notna().any()
    )

    begin_times, acquisition_indices = (
        df[["begin_time", "acquisition_index"]]
        .drop_duplicates()
        .sort_values("begin_time")
        .T.values
    )

    _channels_per_acq = [df[df["begin_time"] == bt]["ch"].unique() for bt in begin_times]
    _z_indices_per_acq = [len(df[df["begin_time"] == bt]["z_index"].unique()) for bt in begin_times]

    if not _all_equal(_channels_per_acq):
        raise ValueError("Acquisitions have different channels and cannot be merged.")
    if not _all_equal(_z_indices_per_acq):
        raise ValueError("Acquisitions have an unequal number of Z-slices and cannot be merged.")

    channels_list = _channels_per_acq[0]
    channels_list = np.sort(channels_list)
    n_channels = len(channels_list)
    n_z_indices = _z_indices_per_acq[0]
    n_time_points = sum(len(df[df["begin_time"] == bt]["time_point"].unique()) for bt in begin_times)

    # Image dimensions from the first record
    first_img_path = df.iloc[0]['tif_path']
    sample_img = _read_tif(first_img_path)
    n_y, n_x = sample_img.shape

    # Build correction once for the whole group — same camera / shading per channel.
    if isinstance(acquisition_metadata, list):
        acq_list = acquisition_metadata
    else:
        acq_list = [acquisition_metadata]
    correction_funcs: dict[Tuple[int, int], object] = {}
    if correct:
        correction_funcs = _build_correction_funcs(
            df, acq_list, begin_times, acquisition_indices, action
        )

    # Validate per-tile or per-field counts.
    if is_tiled_group:
        tile_pairs = (
            df.dropna(subset=["tile_x_index", "tile_y_index"])
              [["tile_x_index", "tile_y_index"]]
              .drop_duplicates()
              .astype(int)
              .itertuples(index=False, name=None)
        )
        tile_pairs = list(tile_pairs)
        n_tiles = len(tile_pairs)
        expected_images = n_time_points * n_channels * n_z_indices * n_tiles
    else:
        n_tiles = 1
        expected_images = n_time_points * n_channels * n_z_indices

    if len(df) != expected_images:
        error_msg = (
            f"DataFrame size mismatch: got {len(df)} records, "
            f"but expected {n_time_points} timepoints × {n_channels} channels × "
            f"{n_z_indices} Z-indices × {n_tiles} tiles = {expected_images}"
        )
        raise AssertionError(error_msg)

    timestamps: list = []
    if is_tiled_group:
        # Build per-tile DataArrays then stitch into a single mosaic.
        tile_arrays: dict[Tuple[int, int], xr.DataArray] = {}
        per_tile_timestamps: list[list] = []
        for (tx, ty) in tile_pairs:
            tile_df = df[
                (df["tile_x_index"] == tx) & (df["tile_y_index"] == ty)
            ]
            tile_arr, ts = _build_tile_array(
                tile_df, correction_funcs, begin_times, acquisition_indices, n_y, n_x
            )
            tile_arrays[(int(tx), int(ty))] = tile_arr
            per_tile_timestamps.append(ts)

        # Use timestamps from the first tile (timepoints are taken roughly together).
        timestamps = per_tile_timestamps[0]

        # Resolve per-timeline overlap from the first acquisition that has a tile spec.
        overlap = 0
        for acq in acq_list:
            ptp = acq.get_partial_tiled_position(int(timeline_index))
            if ptp is not None:
                overlap = int(ptp.overlapping_pixels)
                break
        arr = _stitch_tile_arrays(tile_arrays, overlap_x=overlap, overlap_y=overlap)
    else:
        arr, timestamps = _build_tile_array(
            df, correction_funcs, begin_times, acquisition_indices, n_y, n_x
        )

    arr = _apply_z_projection(arr, current_z_mode)

    arr = arr.squeeze()
    axes = "".join(arr.dims)

    # Pick a reference acquisition for OME metadata.
    first_acq_index = int(acquisition_indices[0])
    if isinstance(acquisition_metadata, list):
        ref_meta = acquisition_metadata[first_acq_index]
    else:
        ref_meta = acquisition_metadata

    channel_list = ref_meta.measurement_setting.channel_list.channel
    # Only consider channels that are actually used by this group; the .mrf
    # MeasurementChannel list can include unused channels acquired with a
    # different objective (different pixel size).
    channel_meta_by_ch = {
        ch.ch: ch for ch in ref_meta.measurement_detail.measurement_channel
    }
    used_meta = [
        channel_meta_by_ch[int(ch_num)]
        for ch_num in channels_list
        if int(ch_num) in channel_meta_by_ch
    ]
    pixel_sizes_x = [m.horizontal_pixel_dimension for m in used_meta]
    pixel_sizes_y = [m.vertical_pixel_dimension for m in used_meta]

    if len(set(pixel_sizes_y)) > 1 or len(set(pixel_sizes_x)) > 1:
        raise ValueError("Pixel sizes differ between channels and cannot be reconciled.")

    physical_size_y = pixel_sizes_y[0] if pixel_sizes_y else 1.0
    physical_size_x = pixel_sizes_x[0] if pixel_sizes_x else 1.0

    ome_metadata = {
        "axes": axes,
        "PhysicalSizeY": physical_size_y,
        "PhysicalSizeX": physical_size_x,
        "PhysicalSizeYUnit": "µm",
        "PhysicalSizeXUnit": "µm",
        "Channels": [],
        "PlateType": ref_meta.measurement_detail.measurement_sample_plate.name,
        "Operator": ref_meta.measurement_detail.operator_name,
        "Timestamps": timestamps,
        "WellID": well_id,
        "ActionIndex": action_index,
        "Action": action,
        "ZMode": current_z_mode,
    }
    if is_tiled_group:
        ome_metadata["PartialTileIndex"] = group_id
        ome_metadata["Stitched"] = True
        ome_metadata["TileGridX"] = len({k[0] for k in tile_arrays.keys()})
        ome_metadata["TileGridY"] = len({k[1] for k in tile_arrays.keys()})
    else:
        ome_metadata["FieldIndex"] = group_id

    if split_acq_idx is not None:
        ome_metadata["AcquisitionIndex"] = int(split_acq_idx)
        ome_metadata["MergedAcquisitions"] = False

    ome_metadata.update(_timeline_metadata(acq_list, acquisition_indices, int(timeline_index)))
    ome_metadata.update(_empirical_time_metadata(timestamps))

    channel_setting_by_ch = {ch.ch: ch for ch in channel_list}
    for ch_num in channels_list:
        ch_meta = channel_setting_by_ch.get(int(ch_num))
        if ch_meta is not None:
            ch_dict = {
                "Name": f"Channel_{ch_num}",
                "Magnification": ch_meta.magnification,
                "Objective": ch_meta.objective,
                "ExposureTime": ch_meta.exposure_time,
                "Acquisition": ch_meta.acquisition,
                "Method": ch_meta.method,
                "LightSource": ch_meta.light_source_name,
                "Fluorophore": ch_meta.fluorophore,
            }
        else:
            ch_dict = {"Name": f"Channel_{ch_num}"}
        ome_metadata["Channels"].append(ch_dict)

    output_directory = Path(out_dir)
    output_directory.mkdir(exist_ok=True)
    magnification = ome_metadata["Channels"][0].get('Magnification', 0)
    prefix = f"{title}_" if title else ""
    rec_suffix = f"_R{int(split_acq_idx):02d}" if split_acq_idx is not None else ""
    if is_tiled_group:
        # Mosaic file naming uses the partial-tile index in place of field.
        fname = (
            f"{prefix}{well_id}_M{int(group_id):02d}_L{timeline_index}_A{action_index}"
            f"_{action}_{magnification}x{rec_suffix}.ome.tif"
        )
    else:
        fname = (
            f"{prefix}{well_id}_F{int(group_id):02d}_L{timeline_index}_A{action_index}"
            f"_{action}_{magnification}x{rec_suffix}.ome.tif"
        )
    destination = output_directory / fname

    if destination.exists() and not overwrite:
        logger.warning("Skipping existing file: %s", destination)
        return

    logger.info("Writing %s (axes=%s, shape=%s)", destination, axes, arr.shape)

    def clean_metadata_for_json(obj):
        """Recursively converts NumPy types to JSON-serializable Python types."""
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

    # tifffile's OME-XML writer only emits a fixed set of standard keys; any
    # other keys are silently dropped. Move our non-standard payload into the
    # OME Image Description as a JSON blob so it actually gets persisted, and
    # keep OME-standard keys (TimeIncrement etc.) at the top level so they
    # render in the proper OME slots.
    OME_STANDARD = {
        "axes",
        "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ",
        "PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit",
        "TimeIncrement", "TimeIncrementUnit",
        "Channels", "Plane", "Description", "Creator", "Name",
    }
    extras = {k: v for k, v in clean_metadata.items() if k not in OME_STANDARD}
    write_meta = {k: v for k, v in clean_metadata.items() if k in OME_STANDARD}
    if extras:
        write_meta["Description"] = json.dumps(extras, default=str)

    if not dry_run:
        if isinstance(arr.data, da.Array):
            arr_computed = arr.compute()
            data_to_write = arr_computed.values
        else:
            data_to_write = arr.values
        tifffile.imwrite(
            destination,
            data_to_write,
            metadata=write_meta,
            bigtiff=True,
            compression=compress,
        )


def _make_fieldstack_groups(
    merged_records_df: pd.DataFrame,
    tile_mode: TileMode,
    no_merge_actions: Optional[List[str]] = None,
) -> List[Tuple[Tuple, pd.DataFrame]]:
    """Group records into work units suitable for ``write_fieldstack``.

    In ``"per-field"`` mode (default) every (row, column, field_index, timeline,
    action_index, action) gets its own group — preserving prior behaviour
    regardless of whether records carry tile indices.

    In ``"stitch"`` mode tiled records are grouped by ``partial_tile_index``
    instead of ``field_index`` so that all tiles in a mosaic land in the same
    work unit. Non-tiled records still group by ``field_index``.

    ``no_merge_actions`` (case-insensitive action names) suppresses merging of
    timepoints across multiple WPI acquisitions for the listed actions. By
    default, repeat WPI runs sharing the same (well, field, timeline, action)
    are concatenated along T — desirable for slow long-term timelapses, but
    wrong for rapid (BF/2D) bursts where each recording should stay its own
    file. When an action matches, ``acquisition_index`` is appended to the
    groupby and propagated as a 7th element of the group key.
    """
    no_merge = {a.strip().lower() for a in (no_merge_actions or []) if a.strip()}

    groups: List[Tuple[Tuple, pd.DataFrame]] = []

    if tile_mode == "stitch" and "tile_x_index" in merged_records_df.columns:
        tiled_mask = merged_records_df["tile_x_index"].notna()
        tiled_df = merged_records_df[tiled_mask]
        non_tiled_df = merged_records_df[~tiled_mask]
    else:
        tiled_df = merged_records_df.iloc[0:0]
        non_tiled_df = merged_records_df

    def _split_by_action(df: pd.DataFrame):
        """Yield (sub_df, split_acq) pairs splitting on the no-merge allowlist."""
        if df.empty:
            return
        if not no_merge:
            yield df, False
            return
        action_lower = df["action"].astype(str).str.lower()
        mask = action_lower.isin(no_merge)
        if mask.any():
            yield df[mask], True
        if (~mask).any():
            yield df[~mask], False

    def _emit(df: pd.DataFrame, group_col: str, split_acq: bool):
        if df.empty:
            return
        cols = ["row", "column", group_col, "timeline_index", "action_index", "action"]
        if split_acq:
            cols.append("acquisition_index")
        gb = df.groupby(cols, sort=False, dropna=True)
        for tup, gdf in gb:
            r, c, gid, tl, ai, act, *rest = tup
            key: Tuple = (int(r), int(c), int(gid), int(tl), int(ai), str(act))
            if split_acq:
                key = key + (int(rest[0]),)
            groups.append((key, gdf))

    for sub, split in _split_by_action(tiled_df):
        _emit(sub, "partial_tile_index", split)
    for sub, split in _split_by_action(non_tiled_df):
        _emit(sub, "field_index", split)

    return groups


def process_fieldstacks_parallel(
    merged_records_df: Union[pd.DataFrame, "pd.core.groupby.generic.DataFrameGroupBy"],
    acquisitions: List['CellVoyagerAcquisition'],
    out_dir: Union[str, Path],
    *,
    title: str | None = None,
    z_mode: str = "maxz",
    z_mode_BF: str = "keep",
    correct: bool = True,
    overwrite: bool = True,
    max_workers: int = 4,
    tile_mode: TileMode = "per-field",
    no_merge_actions: Optional[List[str]] = None,
) -> None:
    """
    Process and write field stacks (or mosaics) to OME-TIFFs in parallel.

    Each field stack is one independent work unit, dispatched to a worker
    thread. A failure inside one work unit is logged and the run continues —
    one bad acquisition does not break the whole batch.

    Parameters
    ----------
    merged_records_df : pandas.DataFrame or DataFrameGroupBy
        Either a flat merged-records DataFrame (preferred) or a pre-built
        DataFrameGroupBy (legacy callers). When a DataFrame is passed, grouping
        is determined by ``tile_mode``.
    acquisitions : List[CellVoyagerAcquisition]
        The list of acquisition metadata objects, indexed by acquisition_index.
    out_dir : Path
        The destination directory for the output OME-TIFF files.
    title : str or None, optional
        Optional prefix for the output filenames.
    z_mode : str, optional
        The Z-projection mode for fluorescence channels, by default "maxz".
    z_mode_BF : str, optional
        The Z-projection mode for brightfield channels, by default "keep".
    correct : bool, optional
        Whether to apply image corrections, by default True.
    overwrite : bool, optional
        Whether to overwrite existing files, by default True.
    max_workers : int, optional
        The number of parallel threads to use, by default 4.
    tile_mode : {"per-field", "stitch"}, optional
        Tile handling. See ``write_fieldstack`` for details.
    no_merge_actions : list of str, optional
        Action names (case-insensitive) for which timepoints should NOT be
        merged across multiple WPI acquisitions. Default: merge everything
        (current behaviour). Use e.g. ``["BF", "2D"]`` to keep each rapid
        recording as its own file — files for split groups gain an
        ``_R{acquisition_index:02d}`` suffix.
    """

    if isinstance(merged_records_df, pd.DataFrame):
        groups = _make_fieldstack_groups(
            merged_records_df, tile_mode, no_merge_actions
        )
    else:
        # Backwards compatibility — already a DataFrameGroupBy.
        groups = list(merged_records_df)

    def _process_single_stack(args):
        key, stack_df = args
        try:
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
                tile_mode=tile_mode,
            )
        except Exception as exc:
            logger.error("Failed to write field stack %s: %s", key, exc, exc_info=True)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(
            tqdm(
                executor.map(_process_single_stack, groups),
                total=len(groups),
                desc="Processing field stacks",
            )
        )
