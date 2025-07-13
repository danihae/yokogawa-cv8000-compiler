from __future__ import annotations

import logging
import re
import sys
from pathlib import Path
from typing import Iterable, Tuple, List, Sequence, Union
from typing import Optional, Any

import numpy as np
import pandas as pd
import xmltodict

from cellvoyager_types import CellVoyagerAcquisition
from yokogawa_cv8000_utils.discovery import path_exists, logger


def read_and_parse_xml(
    path: Union[str, Path],
    model_name: Optional[str] = None
) -> dict[str, Any]:
    """
    Parse an XML file from a plain file-system path.

    Parameters
    ----------
    path : str or Path
        File-system path.
    model_name : str, optional
        Top-level key to unwrap.

    Returns
    -------
    dict
        Parsed (and optionally unwrapped) XML.
    """
    path_str = str(path)
    xml_text = Path(path_str).read_text(encoding="utf-8")

    _dict = xmltodict.parse(
        xml_text,
        process_namespaces=True,
        namespaces={"http://www.yokogawa.co.jp/BTS/BTSSchema/1.0": None},
        attr_prefix="",
        cdata_key="Value",
    )
    return unwrap_model_dict(_dict, model_name=model_name)

def unwrap_model_dict(data: dict, model_name: Optional[str]):
    """
    Return `data[model_name]` if it exists and is a dict.

    Parameters
    ----------
    data : dict
        Dictionary to inspect.
    model_name : str or None
        Key to unwrap.

    Returns
    -------
    dict
        Unwrapped or original dictionary.
    """
    if model_name and model_name in data and isinstance(data[model_name], dict):
        return data[model_name]
    return data


def parse_measurement(wpi_path: Path) -> CellVoyagerAcquisition:
    logger.info(f"⇢  parsing measurement {wpi_path}")
    if not path_exists(wpi_path):
        raise FileNotFoundError(f"{wpi_path} does not exist.")
    wpi_dict = read_and_parse_xml(wpi_path, model_name='WellPlate')
    parent_folder = wpi_path.parent
    mlf_dict = read_and_parse_xml(parent_folder / "MeasurementData.mlf", model_name='MeasurementData')
    mrf_dict = read_and_parse_xml(parent_folder / "MeasurementDetail.mrf", model_name='MeasurementDetail')
    mes_path = (
            wpi_path.parent / mrf_dict["MeasurementSettingFileName"]
    )
    mes_dict = read_and_parse_xml(mes_path, model_name='MeasurementSetting')

    return CellVoyagerAcquisition(
        parent=parent_folder,
        well_plate=wpi_dict,
        measurement_data=mlf_dict,
        measurement_detail=mrf_dict,
        measurement_setting=mes_dict,
    )


def _filter_paths(
    paths: Iterable[Path], exclude: Sequence[str] | str | None = None
) -> list[Path]:
    """Return the paths that do *not* contain any of the exclude keywords."""
    if exclude is None:
        return list(paths)

    if isinstance(exclude, str):
        keywords = [exclude]
    else:
        keywords = list(exclude)

    # Build a single regex that matches any keyword, ignoring case
    pattern = re.compile("|".join(map(re.escape, keywords)), flags=re.IGNORECASE)

    return [p for p in paths if not pattern.search(str(p))]

def _filter_include_paths(
    paths: Iterable[Path], include: Sequence[str] | str | None = None
) -> list[Path]:
    """Return the paths that contain at least one of the include keywords."""
    if include is None:
        return list(paths)

    if isinstance(include, str):
        keywords = [include]
    else:
        keywords = list(include)

    # Build a single regex that matches any keyword, ignoring case
    pattern = re.compile("|".join(map(re.escape, keywords)), flags=re.IGNORECASE)

    return [p for p in paths if pattern.search(str(p))]


def parse_measurements(
    wpi_paths: Union[Path, Iterable[Path]],
    *,
    exclude_keyword: str | Sequence[str] | None = None,
    include_keyword: str | Sequence[str] | None = None,
) -> Tuple[pd.DataFrame, List[CellVoyagerAcquisition]]:
    """
    Parse several Yokogawa CV8000 *WPI* files belonging to the same plate and merge their
    image-level metadata into one table.

    Parameters
    ----------
    wpi_paths
        Iterable with paths to *.wpi* files.
    exclude_keyword
        Keyword or sequence of keywords; every path containing **any** of them
        (case-insensitive) is skipped.  Pass `None` to keep all files.
    include_keyword
        Keyword or sequence of keywords; only paths containing **at least one** of them
        (case-insensitive) are kept.  Pass `None` to skip inclusion filtering.

    Returns
    -------
    (pd.DataFrame, list[CellVoyagerAcquisition])
    """
    if isinstance(wpi_paths, Path):
        wpi_paths = [wpi_paths]

    # Exclude unwanted files first
    wpi_paths = _filter_paths(wpi_paths, exclude_keyword)
    if not wpi_paths:
        raise ValueError("After applying exclusion no *.wpi files remain.")

    # Apply inclusion filtering if specified
    wpi_paths = _filter_include_paths(wpi_paths, include_keyword)
    if not wpi_paths:
        raise ValueError("After applying inclusion no *.wpi files remain.")

    wpi_paths = list(wpi_paths)  # allow generators
    if not wpi_paths:
        raise ValueError("No *.wpi files provided to parse_measurements")

    acquisitions: List[CellVoyagerAcquisition] = []
    for path in wpi_paths:
        if not path_exists(path):
            raise FileNotFoundError(path)
        acquisitions.append(parse_measurement(path))

    # Merge image-level tables
    merged_records: list[pd.DataFrame] = []
    for acq_idx, (path, acq) in enumerate(zip(wpi_paths, acquisitions)):
        records = acq.get_image_measurement_records()
        df = pd.DataFrame.from_records(m.model_dump() for m in records)

        # Append additional context columns with metadata
        title = acq.measurement_detail.title
        begin_time = acq.measurement_detail.begin_time

        # Append additional context columns
        df["acquisition_index"] = acq_idx
        df["base_path"] = path.parent
        df["tif_path"] = df["value"].apply(lambda rel: path.parent / rel)
        df["title"] = title
        df["begin_time"] = begin_time

        merged_records.append(df)

    merged_records_df = pd.concat(merged_records, ignore_index=True, sort=False)

    return merged_records_df, acquisitions
