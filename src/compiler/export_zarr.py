"""OME-NGFF v0.4 (.ome.zarr) writer for yokogawa-cv8000-compiler.

Sibling of the OME-TIFF writer in ``export.py``. Both consume the same
``ome_metadata`` dict produced inside ``write_fieldstack``; this module maps
that dict onto the NGFF spec:

- Standard OME fields (PhysicalSizeX/Y, TimeIncrement, Channels) become
  ``multiscales[0].coordinateTransformations`` and ``omero.channels``.
- All non-OME-standard custom fields (WellID, FieldIndex/PartialTileIndex,
  Timestamps, RelativeTimes, …) live under a top-level ``cv8000`` namespace
  in ``.zattrs`` — Zarr attrs are JSON-native, so no Description-JSON
  workaround is needed.

The ``zarr`` and ``ome_zarr`` packages are optional extras; this module
imports them lazily so the TIFF-only path stays dependency-free.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import dask
import dask.array as da
import numpy as np


# Mirrors the OME_STANDARD set inside write_fieldstack: any key NOT in here
# is treated as a custom yokogawa field and routed to the cv8000 namespace
# of .zattrs. Keep this in sync with export.py if new OME-standard keys are
# added there.
_OME_STANDARD_KEYS = {
    "axes",
    "PhysicalSizeX", "PhysicalSizeY", "PhysicalSizeZ",
    "PhysicalSizeXUnit", "PhysicalSizeYUnit", "PhysicalSizeZUnit",
    "TimeIncrement", "TimeIncrementUnit",
    "Channels", "Plane", "Description", "Creator", "Name",
}

_AXIS_SPEC = {
    "t": {"name": "t", "type": "time", "unit": "second"},
    "c": {"name": "c", "type": "channel"},
    "z": {"name": "z", "type": "space", "unit": "micrometer"},
    "y": {"name": "y", "type": "space", "unit": "micrometer"},
    "x": {"name": "x", "type": "space", "unit": "micrometer"},
}


def _clean_for_json(obj: Any) -> Any:
    """Recursively coerce NumPy scalars/arrays to JSON-serializable Python types."""
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(item) for item in obj]
    if isinstance(obj, tuple):
        return [_clean_for_json(item) for item in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _axes_for(axes_str: str) -> list[dict]:
    return [_AXIS_SPEC[c.lower()] for c in axes_str]


def _scale_for(axes_str: str, ome_metadata: dict) -> list[float]:
    """Build the level-0 ``scale`` transform aligned to ``axes_str``.

    Pixel sizes for missing dims default to 1.0; TimeIncrement defaults to
    1.0 if not measured (single-timepoint or unparseable timestamps).
    """
    scale_map = {
        "t": float(ome_metadata.get("TimeIncrement") or 1.0),
        "c": 1.0,
        "z": float(ome_metadata.get("PhysicalSizeZ") or 1.0),
        "y": float(ome_metadata.get("PhysicalSizeY") or 1.0),
        "x": float(ome_metadata.get("PhysicalSizeX") or 1.0),
    }
    return [scale_map[c.lower()] for c in axes_str]


def _chunks_for(axes_str: str, shape: tuple[int, ...], yx_chunk: int = 512) -> tuple[int, ...]:
    """Default chunking: 1 along T/C/Z, ``yx_chunk`` along Y/X (clipped to shape)."""
    chunks: list[int] = []
    for axis_letter, dim in zip(axes_str, shape):
        if axis_letter.lower() in ("y", "x"):
            chunks.append(min(yx_chunk, dim))
        else:
            chunks.append(1)
    return tuple(chunks)


def _decimate_yx(data, axes_str: str, factor: int):
    """Stride-decimate along Y and X by ``factor`` — nearest-neighbour 2×^k downsample.

    Preserves dtype (no float upcast); cheap on dask arrays (slicing).
    """
    if factor == 1:
        return data
    slices = []
    for letter in axes_str:
        if letter.lower() in ("y", "x"):
            slices.append(slice(None, None, factor))
        else:
            slices.append(slice(None))
    return data[tuple(slices)]


def _percentile_windows(coarsest_np: np.ndarray, axes_str: str) -> list[tuple[float, float]]:
    """Per-channel 1st/99th percentile windows from the coarsest pyramid level.

    Falls back to whole-array percentiles when there's no channel axis.
    """
    c_idx = axes_str.lower().find("c")
    if c_idx == -1:
        p1, p99 = np.percentile(coarsest_np, (1, 99))
        return [(float(p1), float(p99))]
    moved = np.moveaxis(coarsest_np, c_idx, 0)
    moved = moved.reshape(moved.shape[0], -1)
    p1 = np.percentile(moved, 1, axis=1)
    p99 = np.percentile(moved, 99, axis=1)
    return list(zip([float(v) for v in p1], [float(v) for v in p99]))


def _omero_channels(
    ome_metadata: dict,
    dtype: np.dtype,
    windows: list[tuple[float, float]] | None = None,
) -> list[dict]:
    """Build the ``omero.channels`` block.

    ``windows`` (optional) supplies per-channel ``(start, end)`` from
    percentile estimation; ``min``/``max`` always span the full dtype
    range. When ``windows`` is None (Phase A or non-multiscale), start/end
    fall back to the dtype range.
    """
    info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    win_min = float(info.min) if info is not None else 0.0
    win_max = float(info.max) if info is not None else 1.0

    out: list[dict] = []
    channels = ome_metadata.get("Channels", []) or []
    for i, ch in enumerate(channels):
        if windows is not None and i < len(windows):
            start, end = windows[i]
        else:
            start, end = win_min, win_max
        out.append({
            "label": ch.get("Name", "Channel"),
            "color": "FFFFFF",
            "window": {
                "min": win_min,
                "max": win_max,
                "start": start,
                "end": end,
            },
            "active": True,
        })
    return out


def write_fieldstack_zarr(
    *,
    arr,
    axes_str: str,
    ome_metadata: dict,
    destination: Path,
    overwrite: bool,
    dry_run: bool,
    pyramid_levels: int = 4,
    yx_chunk: int = 512,
) -> None:
    """Write a single fieldstack as an OME-NGFF v0.4 ``.ome.zarr`` directory.

    Parameters
    ----------
    arr : xarray.DataArray
        Squeezed array with axes per ``axes_str``. Backed by a dask array
        (``arr.data``) when constructed by ``write_fieldstack``.
    axes_str : str
        Uppercase axes string from the squeezed array (e.g. "TCZYX", "CYX").
    ome_metadata : dict
        The full OME metadata dict assembled by ``write_fieldstack``.
        OME-standard keys map to NGFF transforms / omero block; the rest
        lands under ``cv8000`` in ``.zattrs``.
    destination : Path
        Target directory ending in ``.ome.zarr``. Created if missing.
    overwrite : bool
        If True and the directory exists, remove it first.
    dry_run : bool
        If True, build metadata but skip all I/O.
    pyramid_levels : int, optional
        Number of resolution levels to write (default 4: full + 3 downsampled).
        ``1`` writes only the full-resolution dataset. Downsampling is
        2× per level along Y and X only (T/C/Z untouched), nearest-style
        stride decimation — preserves uint16 dtype.
    yx_chunk : int, optional
        Chunk size along Y and X (default 512). T/C/Z chunked at 1.
    """
    try:
        import zarr
        from numcodecs import Blosc
    except ImportError as e:
        raise ImportError(
            "OME-Zarr output requires the [zarr] extra: "
            "uv pip install -e '.[zarr]'"
        ) from e

    destination = Path(destination)
    if destination.exists():
        if not overwrite:
            return
        if not dry_run:
            shutil.rmtree(destination)

    shape = tuple(arr.shape)
    chunks = _chunks_for(axes_str, shape, yx_chunk=yx_chunk)
    axes_meta = _axes_for(axes_str)
    base_scale = _scale_for(axes_str, ome_metadata)

    cv8000_extras = _clean_for_json({
        k: v for k, v in ome_metadata.items() if k not in _OME_STANDARD_KEYS
    })
    dtype = np.dtype(arr.dtype)

    if dry_run:
        return

    # Decide actual level count: stop when YX would drop below 16 pixels.
    yi = axes_str.lower().find("y")
    xi = axes_str.lower().find("x")
    actual_levels = 1
    if pyramid_levels > 1 and yi >= 0 and xi >= 0:
        for level in range(1, pyramid_levels):
            stride = 2 ** level
            y_size = shape[yi] // stride
            x_size = shape[xi] // stride
            if y_size < 16 or x_size < 16:
                break
            actual_levels = level + 1

    # Pull the dask array out of the xarray wrapper.
    data = (
        arr.data if isinstance(arr.data, da.Array)
        else da.from_array(np.asarray(arr.data), chunks=chunks)
    )

    store = zarr.DirectoryStore(str(destination))
    zarr.group(store=store, overwrite=True)
    compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)

    datasets_meta: list[dict] = []
    # Force synchronous scheduling — outer ThreadPoolExecutor already
    # parallelizes across fieldstacks; default "threads" would nest pools.
    with dask.config.set(scheduler="synchronous"):
        for level in range(actual_levels):
            stride = 2 ** level
            level_data = _decimate_yx(data, axes_str, stride)

            # Rechunk per-level to keep YX chunks at yx_chunk after decimation
            # (decimated array would otherwise inherit too-large chunk hints).
            level_chunks = _chunks_for(axes_str, tuple(level_data.shape), yx_chunk=yx_chunk)
            if level_data.chunksize != level_chunks:
                level_data = level_data.rechunk(level_chunks)

            da.to_zarr(
                level_data,
                url=store,
                component=str(level),
                overwrite=True,
                compressor=compressor,
                compute=True,
                return_stored=False,
            )

            level_scale = list(base_scale)
            for i, letter in enumerate(axes_str):
                if letter.lower() in ("y", "x"):
                    level_scale[i] = base_scale[i] * stride
            datasets_meta.append({
                "path": str(level),
                "coordinateTransformations": [
                    {"type": "scale", "scale": level_scale},
                ],
            })

    # Compute display windows from the coarsest level (cheap; small array).
    coarsest = zarr.open(str(destination), mode="r")[str(actual_levels - 1)][:]
    windows = _percentile_windows(coarsest, axes_str)

    omero_block = {
        "version": "0.4",
        "channels": _omero_channels(ome_metadata, dtype, windows=windows),
    }
    multiscales = [{
        "version": "0.4",
        "name": destination.stem,
        "axes": axes_meta,
        "datasets": datasets_meta,
    }]
    zattrs = {
        "multiscales": multiscales,
        "omero": omero_block,
        "cv8000": cv8000_extras,
    }

    root = zarr.open(str(destination), mode="r+")
    root.attrs.update(zattrs)
