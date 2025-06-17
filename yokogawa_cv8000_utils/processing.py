from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Sequence
import logging, io, zipfile, numpy as np

import dask.array as da
import xarray as xr
import tifffile

# pydantic models shipped by cellvoyager-types
from cellvoyager_types._metadata import CellVoyagerAcquisition   # <- high-level helper
from cellvoyager_types._xarray import dataarray_from_metadata                  # xarray builder

# local helper that already understands “zip!internal/file.tif”
from .discovery import read_file_content, is_zip_measurement

LOGGER = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# small container holding one FoV                                             #
# --------------------------------------------------------------------------- #
@dataclass
class FieldStack:
    data: xr.DataArray            # dims = (t, a, c, z, y, x)
    timestamps: np.ndarray        # DeltaT (s) per time plane


# --------------------------------------------------------------------------- #
# internal – open a 2-D plane transparently                                  #
# --------------------------------------------------------------------------- #
def _imread_virtual(path: str | Path) -> np.ndarray:
    p = str(path)
    if "!" in p:
        zip_path, internal = p.split("!", 1)
        with zipfile.ZipFile(zip_path) as z:
            with z.open(internal) as fh:
                with tifffile.TiffFile(io.BytesIO(fh.read())) as tf:
                    return tf.asarray()
    return tifffile.imread(p)


# --------------------------------------------------------------------------- #
# 1.  parse ONE measurement folder (or ZIP)                                   #
# --------------------------------------------------------------------------- #
def parse_measurement(meas_dir: Path) -> Dict[Tuple[str, int], FieldStack]:
    """
    Parameters
    ----------
    meas_dir
        Directory that contains the three Yokogawa XML files *or*
        'archive.zip!path/MeasurementData.mlf'.

    Returns
    -------
    {(well_id, field_index): FieldStack}
    """
    LOGGER.info(f"⇢  parsing measurement {meas_dir}")
    # ------------------------------------------------------------------- #
    # get XML strings                                                     #
    # ------------------------------------------------------------------- #
    if "!" in str(meas_dir):                                           # inside ZIP
        zip_path, internal = str(meas_dir).split("!", 1)
        base = Path(zip_path)
        base_inside = Path(internal).parent
        xml_mlf = read_file_content(Path(f"{zip_path}!{base_inside/'MeasurementData.mlf'}"))
        xml_mrf = read_file_content(Path(f"{zip_path}!{base_inside/'MeasurementResult.mrf'}"))
        xml_set = read_file_content(Path(f"{zip_path}!{base_inside/'MeasurementSetting.xml'}"))
        parent_folder = Path(f"{zip_path}!{base_inside}")              # virtual root
    else:                                                              # normal dir
        xml_mlf = (meas_dir / "MeasurementData.mlf").read_text()
        xml_mrf = (meas_dir / "MeasurementResult.mrf").read_text()
        xml_set = (meas_dir / "MeasurementSetting.xml").read_text()
        parent_folder = meas_dir

    # build CellVoyagerAcquisition object – the class collects all XML pieces[1]
    cva = CellVoyagerAcquisition.model_validate_xml(
        measurement_data_xml=xml_mlf,
        measurement_detail_xml=xml_mrf,
        measurement_setting_xml=xml_set,
        parent=str(parent_folder),
    )

    image_records = cva.get_image_measurement_records()                # List[ImageMeasurementRecord][1]

    # bucket → {(row,col,field) : [record, …]}
    rec_by_fov: Dict[Tuple[int, int, int], List] = {}
    for rec in image_records:
        rec_by_fov.setdefault((rec.row, rec.column, rec.field_index), []).append(rec)

    fov_stacks: Dict[Tuple[str, int], FieldStack] = {}

    letters = "ABCDEFGHIJKLMNOP"                                       # e.g. row 1 ➜ 'A'
    for (row, col, field), recs in rec_by_fov.items():
        well_id = f"{letters[row-1]}{col:02}"
        # keep deterministic order
        recs.sort(key=lambda r: (r.time_point, r.action_index, r.ch, r.z_index))

        # build lazy DataArray via the helper shipped in cellvoyager-types
        darr: xr.DataArray = dataarray_from_metadata(                  # dims=(t,a,c,z,y,x)
            parent_folder=parent_folder,
            image_records=recs,
            detail=cva.measurement_detail,
        )

        # collect Δt per time point
        # measurement_detail.time_point_count gives count; the actual
        # delta_t is inside measurement_setting.timelapse.timeline[…]
        delta_ts = _extract_delta_t(cva, darr.sizes["t"])

        fov_stacks[(well_id, field)] = FieldStack(darr, delta_ts)

    return fov_stacks


def _extract_delta_t(cva: CellVoyagerAcquisition, size_t: int) -> np.ndarray:
    """
    Return delta-t (seconds) vector length *size_t* using the timeline
    information from MeasurementSetting.xml.
    """
    # first (and usually only) timeline
    tl = cva.measurement_setting.timelapse.timeline[0]                 # type: Timeline[1]
    # period is in milliseconds in Yokogawa XML
    periodic_s = tl.period / 1000.0
    return np.arange(size_t, dtype=np.float32) * periodic_s


# --------------------------------------------------------------------------- #
# 2.  concatenate several FieldStacks (multi-day)                             #
# --------------------------------------------------------------------------- #
def concat_fieldstacks(stacks: Sequence[FieldStack]) -> FieldStack:
    if len(stacks) == 1:
        return stacks[0]
    data = xr.concat([fs.data for fs in stacks], dim="t")
    tvec = np.concatenate([fs.timestamps for fs in stacks])
    return FieldStack(data, tvec)


# --------------------------------------------------------------------------- #
# 3.  driver that merges *many* measurements                                  #
# --------------------------------------------------------------------------- #
def compile_measurements(measurement_roots: List[Path]) -> Dict[Tuple[str, str, int], FieldStack]:
    """
    Parameters
    ----------
    measurement_roots
        List of directories or “zip!internal” paths that point to individual
        CV8000 measurements of (possibly) the same plate.

    Returns
    -------
    {(plate_barcode, well_id, field_index): FieldStack}
    """
    bucket: Dict[Tuple[str, str, int], List[FieldStack]] = {}

    for root in measurement_roots:
        fov_stacks = parse_measurement(root)

        # read barcode once per measurement
        xml_mlf = read_file_content(root / "MeasurementData.mlf" if "!" not in str(root)
                                    else root)
        barcode = Mlf.model_validate_xml(xml_mlf).measurement_detail.title   # pick what you regard as barcode

        for (well_id, field_idx), fs in fov_stacks.items():
            bucket.setdefault((barcode, well_id, field_idx), []).append(fs)

    # concat multi-day series
    return {key: concat_fieldstacks(lst) for key, lst in bucket.items()}
