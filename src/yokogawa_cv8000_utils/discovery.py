import logging
from pathlib import Path
from typing import Iterator, Union

logger = logging.getLogger(__name__)

def find_measurements(root: Path) -> Iterator[Path]:
    """
    Find all CV8000 measurements in a directory tree.

    Searches for <PlateName>.wpi files in regular directories.

    Parameters
    ----------
    root : Path
        Root directory to search.

    Yields
    ------
    Path
        Path objects pointing to .wpi files.
    """
    logger.info(f"Searching for measurements in {root}")

    # Find measurements in directories
    yield from root.rglob("*.wpi")

def get_measurement_directory(path: Path) -> Path:
    """
    Get the directory containing a measurement.

    For regular measurements, returns the parent directory.

    Parameters
    ----------
    path : Path
        Path to measurement file.

    Returns
    -------
    Path
        Directory containing the measurement.
    """
    return path.parent

def path_exists(path: Union[Path, str]) -> bool:
    """
    Return True if *path* exists.

    Works for normal filesystem paths.

    Parameters
    ----------
    path : Path or str
        Path to check.

    Returns
    -------
    bool
        True if the path exists, False otherwise.
    """
    return Path(str(path)).exists()
