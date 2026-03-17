# Yokogawa CV8000 Utils

A set of tools for the UMG Pharmacology microscopy workflow:

1. **CV8000 Data Compiler** — CLI tool that reads Yokogawa CV8000 output, compiles multi-channel / multi-timepoint / multi-Z data into TIFF stacks, and applies z-projections
2. **Microscopy Experiment Manager** — Gradio web app for registering experiments, creating metadata-tagged folders on the NAS, and managing multi-day acquisitions

## CV8000 Data Compiler

A Python tool for processing output data from Yokogawa Cell Voyager CV8000 high-content screening systems. It discovers measurement files, parses their metadata, applies dark-field and flat-field corrections, and writes compiled image stacks as OME-TIFF files — no proprietary Yokogawa software required.

### Features

- Discovers `.wpi` measurement files recursively within a directory tree
- Parses `MeasurementData.mlf`, `MeasurementDetail.mrf`, and `.mes` metadata files
- Applies dark-field and flat-field (shading) corrections automatically
- Compiles multi-channel, multi-timepoint, and multi-Z data into 5D image stacks
- Multiple Z-projection modes for fluorescence and brightfield channels
- Writes well-annotated OME-TIFF files with embedded physical pixel sizes and channel metadata
- Parallel processing via `ThreadPoolExecutor`

### Installation

Requires Python ≥ 3.10.

Clone the repository and install with pip (or [uv](https://github.com/astral-sh/uv)):

```bash
git clone https://github.com/yourusername/yokogawa-utils.git
cd yokogawa-utils

# with pip
pip install -e .

# or with uv
uv sync
```

## Usage

After installation, the `compile-cv8000` command is available on your PATH.

### Basic usage

```bash
compile-cv8000 --root-dir /path/to/data --out-dir /path/to/output
```

### All options

```text
compile-cv8000 [OPTIONS]

Options:
  --root-dir PATH        Root directory to search for .wpi measurement files [required]
  --out-dir PATH         Directory where OME-TIFF files will be saved [required]
  --title TEXT           Base name for output files (default: compiled_data)
  --z-mode TEXT          Z-projection mode for fluorescence channels (default: maxz)
  --exclude TEXT         Keyword to exclude measurements whose path contains this string
  --overwrite / --no-overwrite
                         Overwrite existing output files (default: True)
  --max-workers INTEGER  Number of parallel workers (default: CPU count)
  --help                 Show this message and exit.
```

### Z-projection modes

| Mode          | Description                                                            |
|---------------|------------------------------------------------------------------------|
| `keep`        | Keep all Z-slices (no projection)                                      |
| `mip`         | Maximum intensity projection                                           |
| `maxz`        | Select the single Z-slice with the highest mean intensity (default)    |
| `osbm`        | Out-of-focus-Suppressed Brightfield Maximum (gradient-based)           |
| `max_entropy` | Select the Z-slice with the highest Shannon entropy                    |
| `min_entropy` | Select the Z-slice with the lowest Shannon entropy                     |

Brightfield channels (`BF3D` action) use a separate `--z-mode-bf` setting (not exposed in the CLI; defaults to `keep`).

### Examples

```bash
# Compile with MIP projection, 8 workers, exclude a test plate
compile-cv8000 \
  --root-dir /data/experiments/plate_run_01 \
  --out-dir /data/compiled/plate_run_01 \
  --title plate_run_01 \
  --z-mode mip \
  --max-workers 8 \
  --exclude test

# Dry-run preview (run via Python API, see below)
```

### Output files

Each field stack is written as a single OME-TIFF:

```text
{title}_{WellID}_F{field:02d}_L{timeline}_A{action_index}_{action}_{magnification}x.ome.tif
```

For example:

```text
compiled_data_B03_F01_L0_A1_Fluorescence_20x.ome.tif
```

Files embed OME metadata including physical pixel sizes (µm), channel names, timestamps, operator, plate type, well ID, and Z-projection mode.

## Python API

The package can also be used programmatically:

```python
from pathlib import Path
from yokogawa_cv8000_utils.discovery import find_measurements
from yokogawa_cv8000_utils.processing import parse_measurements
from yokogawa_cv8000_utils.export import process_fieldstacks_parallel

root_dir = Path("/path/to/data")
out_dir = Path("/path/to/output")

# 1. Find .wpi files
wpi_paths = list(find_measurements(root_dir))

# 2. Parse metadata and image records
merged_df, acquisitions = parse_measurements(wpi_paths, exclude_keyword="test")

# 3. Group into per-field stacks
group_cols = ["row", "column", "field_index", "timeline_index", "action_index", "action"]
fieldstacks = merged_df.groupby(group_cols, sort=False, dropna=True)

# 4. Write OME-TIFFs in parallel
process_fieldstacks_parallel(
    fieldstacks,
    acquisitions,
    out_dir,
    title="my_experiment",
    z_mode="maxz",
    max_workers=8,
)
```

## Dependencies

- `dask` — lazy parallel array computation
- `numpy` — numerical operations
- `pandas` — tabular metadata management
- `pydantic>=2` — metadata model validation
- `tifffile` — reading and writing TIFF/OME-TIFF files
- `tqdm` — progress bars
- `typer` — CLI interface
- `xarray` — labelled N-dimensional arrays
- `xmltodict` — XML metadata parsing

## Microscopy Experiment Manager (Web GUI)

A Gradio web app for registering new microscopy experiments. Researchers fill in metadata before acquisition; the app creates a deterministic hash-named folder on the NAS and returns the path to paste into the microscopy software.

### Key capabilities

- **New Experiment** tab — fill in metadata, create a folder, get a copyable path
- **Continue Experiment** tab — browse and resume previous experiments (sorted by date, with full metadata display)
- **Multi-day measurements** — mark experiments that span multiple acquisition sessions
- **Filesystem-based storage** — each experiment is a self-contained folder with its own `metadata.json` (no central database to corrupt)
- **Deterministic folder names** — identical metadata on the same day always produces the same folder

### Quick start (Docker)

1. Copy the environment template and edit it:

   ```bash
   cp .env.example .env
   # Edit .env — set UPLOAD_ROOT to the NAS upload path
   ```

2. Build and start the services:

   ```bash
   docker compose up -d
   ```

The web GUI is available at `http://<host>:80` (via nginx) or directly at `http://<host>:7860`.

### Configuration (.env)

| Variable | Description | Default |
| --- | --- | --- |
| `UPLOAD_ROOT` | Path where experiment folders are created | `/volume1/upload` |
| `UPLOAD_ROOT_DISPLAY` | Path shown to the user (only needed in Docker, when different from `UPLOAD_ROOT`) | same as `UPLOAD_ROOT` |
| `GRADIO_PORT` | Port for the Gradio server | `7860` |

### How it works

1. Researcher opens the web GUI on the acquisition computer
2. Fills in required metadata (user, project, instrument, modality, cell type, well type)
3. Clicks "Create Experiment Folder"
4. App creates a hash-named folder on the NAS with a `metadata.json` inside
5. Folder is made writable so the microscopy software can save data
6. Researcher copies the returned path into the microscopy software
7. For multi-day experiments, the "Continue Experiment" tab lists all previous experiments for easy resumption

## Code Structure

- **`yokogawa_cv8000_compiler/`** — CLI tool for compiling CV8000 image data
  - `__main__.py` — CLI entry point
  - `compiling.py` — projection and compilation logic
  - `parsing.py` — metadata file parsing
  - `utils.py` — helper functions
- **`web/`** — Gradio web app for experiment registration
  - `app.py` — Gradio application
  - `Dockerfile` — container image
  - `nginx.conf` — reverse proxy config
- **`docker-compose.yml`** — orchestrates Gradio + nginx services

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
