"""
export.py
=========

Write FieldStack objects (see processing.py) to OME-TIFF.

• One file per action_index  (A000, A001, …)
• Z handling modes
      "keep"   – keep full Z dimension
      "mip"    – maximum-intensity projection over Z
      "maxz"   – pick single slice with highest overall intensity
• tifffile is used as backend; a minimal OME header is produced via the
  metadata={'axes': 'TCZYX', ...} keyword[2][3].

Example
-------
from pathlib import Path
from ycvcompiler.export import write_fieldstack
key = ("PLT42", "A05", 3)        # (plate, well, field_idx)
write_fieldstack(key, fs, Path("out"))
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Tuple

import numpy as np
import tifffile  # writes OME-TIFF[2][3][4]

from .processing import FieldStack

LOGGER = logging.getLogger(__name__)


# -----------------------------------------------------------------------------#
# public API                                                                   #
# -----------------------------------------------------------------------------#
def write_fieldstack(
    key: Tuple[str, str, int],
    fs: FieldStack,
    out_dir: Path,
    *,
    z_mode: Literal["keep", "mip", "maxz"] = "keep",
    compress: str | None = "zstd",
    overwrite: bool = False,
) -> None:
    """
    Parameters
    ----------
    key
        (plate_barcode, well_id, field_index)
    fs
        FieldStack from processing.compile_measurements()
        dims = ('t','a','c','z','y','x')
    out_dir
        Destination folder (is created if necessary)
    z_mode
        • "keep"  – keep Z, output axes 'TCZYX'
        • "mip"   – max-intensity projection,   axes 'TCYX'
        • "maxz"  – choose single Z slice,      axes 'TCYX'
    compress
        tifffile compression algorithm, e.g. "zstd", "lzma", None
    overwrite
        Replace existing files
    """
    plate, well, field = key
    out_dir.mkdir(parents=True, exist_ok=True)

    # iterate over action axis
    for ai, action in enumerate(fs.data.coords.get("a").values):
        da = fs.data.sel(a=action)                       # dims = t,c,z,y,x

        if z_mode == "mip":                              # MIP over z
            da = da.max(dim="z")
            axes = "TCYX"
        elif z_mode == "maxz":                           # slice with global max
            z_sel = int(da.max(("t", "c", "y", "x")).argmax(dim="z").data)
            da = da.isel(z=z_sel).squeeze("z")
            axes = "TCYX"
        else:                                            # keep full Z stack
            axes = "TCZYX"

        # tifffile wants numpy with axis order matching 'axes'
        order = tuple(axes.lower())                      # e.g. ('t','c','y','x')
        arr = da.transpose(*order).data                  # numpy/dask -> numpy
        arr = np.asarray(arr)                            # ensure concrete

        # build output path
        fname = f"{plate}_{well}_F{field:02d}_A{ai:02d}.ome.tif"
        dest = out_dir / fname
        if dest.exists() and not overwrite:
            LOGGER.warning("skip existing %s", dest)
            continue

        LOGGER.info("write %s (%s)", dest, axes)
        tifffile.imwrite(                                # minimal OME writer[2]
            dest,
            arr,
            metadata={"axes": axes},
            bigtiff=True,
            compression=compress,
        )

# -----------------------------------------------------------------------------#
# convenience batch helper                                                     #
# -----------------------------------------------------------------------------#
def write_plate(
    stacks: dict[Tuple[str, str, int], FieldStack],
    out_dir: Path,
    *,
    z_mode: Literal["keep", "mip", "maxz"] = "keep",
    workers: int | None = None,
    **kw,
):
    """
    Write every FieldStack of a plate dictionary returned by
    processing.compile_measurements().

    Executes in parallel via ProcessPoolExecutor if `workers` > 1.
    """
    if workers is None or workers <= 1:
        for k, fs in stacks.items():
            write_fieldstack(k, fs, out_dir, z_mode=z_mode, **kw)
    else:
        import concurrent.futures as cf

        with cf.ProcessPoolExecutor(max_workers=workers) as pool:
            fut = {
                pool.submit(write_fieldstack, k, fs, out_dir,
                            z_mode=z_mode, **kw): k for k, fs in stacks.items()
            }
            for f in cf.as_completed(fut):
                try:
                    f.result()
                except Exception:
                    LOGGER.exception("failed %s", fut[f])
