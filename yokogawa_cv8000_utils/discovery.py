"""
Discovery module for finding Yokogawa CV8000 measurements.
Handles both regular directories and ZIP archives.
"""
from pathlib import Path
from typing import Iterator, Union
import zipfile
import logging

logger = logging.getLogger(__name__)


def find_measurements(root: Path) -> Iterator[Path]:
    """
    Find all CV8000 measurements in a directory tree.

    Searches for MeasurementData.mlf files in both regular directories
    and ZIP archives. Returns virtual paths for ZIP contents using
    the format: /path/to/archive.zip!internal/path/MeasurementData.mlf

    Args:
        root: Root directory to search

    Yields:
        Path objects pointing to MeasurementData.mlf files
    """
    logger.info(f"Searching for measurements in {root}")

    # Find measurements in regular directories
    regular_files = list(root.rglob("*MeasurementData.mlf"))
    logger.debug(f"Found {len(regular_files)} regular measurement files")

    for mlf_file in regular_files:
        yield mlf_file

    # Find measurements in ZIP archives
    zip_count = 0
    for zip_path in root.rglob("*.zip"):
        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                for internal_path in zf.namelist():
                    if internal_path.endswith("MeasurementData.mlf"):
                        # Create virtual path using ! separator
                        virtual_path = Path(f"{zip_path}!{internal_path}")
                        logger.debug(f"Found measurement in ZIP: {virtual_path}")
                        zip_count += 1
                        yield virtual_path
        except (zipfile.BadZipFile, PermissionError) as e:
            logger.warning(f"Could not read ZIP file {zip_path}: {e}")

    logger.info(f"Found {zip_count} measurements in ZIP archives")


def read_file_content(path: Union[Path, str]) -> str:
    """
    Read content from either regular files or files within ZIP archives.

    Args:
        path: Path to file, or virtual path for ZIP contents

    Returns:
        File content as string

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If ZIP path format is invalid
    """
    path_str = str(path)

    if '!' in path_str:
        # Handle ZIP archive virtual path
        zip_path, internal_path = path_str.split('!', 1)
        zip_path = Path(zip_path)

        if not zip_path.exists():
            raise FileNotFoundError(f"ZIP archive not found: {zip_path}")

        try:
            with zipfile.ZipFile(zip_path, 'r') as zf:
                with zf.open(internal_path) as f:
                    return f.read().decode('utf-8')
        except KeyError:
            raise FileNotFoundError(f"File not found in ZIP: {internal_path}")
        except zipfile.BadZipFile:
            raise ValueError(f"Invalid ZIP file: {zip_path}")
    else:
        # Handle regular file
        return Path(path).read_text(encoding='utf-8')


def is_zip_measurement(path: Path) -> bool:
    """
    Check if a path represents a measurement within a ZIP archive.

    Args:
        path: Path to check

    Returns:
        True if path is a ZIP virtual path
    """
    return '!' in str(path)


def get_measurement_directory(path: Path) -> Path:
    """
    Get the directory containing a measurement.

    For ZIP measurements, returns the ZIP file path.
    For regular measurements, returns the parent directory.

    Args:
        path: Path to measurement file

    Returns:
        Directory or ZIP file containing the measurement
    """
    if is_zip_measurement(path):
        zip_path = str(path).split('!')[0]
        return Path(zip_path)
    else:
        return path.parent
