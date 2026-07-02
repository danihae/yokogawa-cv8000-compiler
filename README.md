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
- Optional **OME-Zarr** (OME-NGFF v0.4) output, alongside or instead of OME-TIFF
- Desktop **GUI** (Tkinter) front-end and a static plate-map **explorer** for browsing results
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

OME-Zarr output is optional. To enable the `--format zarr`/`both` modes, install
the `[zarr]` extra:

```bash
pip install -e '.[zarr]'   # or: uv sync --extra zarr
```

MP4 previews (see [MP4 previews](#mp4-previews)) need the `[video]` extra, which
bundles a static ffmpeg — no system ffmpeg required:

```bash
pip install -e '.[video]'   # or: uv sync --extra video
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
  --z-mode-bf TEXT       Z-projection mode for brightfield (BF3D) channels;
                         one of: keep, osbm (default: keep)
  --tile-mode TEXT       How to handle tiled acquisitions: 'per-field' (one
                         OME-TIFF per FieldIndex, default) or 'stitch' (blend
                         each PartialTileIndex grid into one mosaic file)
  --no-merge-actions TEXT
                         Comma-separated action names (e.g. 'BF,2D') for
                         which timepoints should NOT be merged across
                         multiple WPI acquisitions. Useful for rapid
                         (~20 Hz) bursts. Files for split groups gain
                         _R{idx:02d}. Default: merge everything.
  --format TEXT          Output format: 'tiff' (OME-TIFF, default), 'zarr'
                         (OME-NGFF v0.4 .ome.zarr), or 'both'. The zarr and
                         both modes require the [zarr] extra (see Installation).
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

Brightfield channels (`BF3D` action) use the separate `--z-mode-bf` setting, which accepts `keep` or `osbm` and defaults to `keep`. `osbm` is the recommended choice for collapsing brightfield Z-stacks into a single sharp slice.

### Tiled acquisitions

When the original acquisition uses Yokogawa's `PartialTiledPosition` (each
field is one tile of a larger grid), `--tile-mode` controls the output:

| Mode         | Behaviour                                                                                          |
|--------------|----------------------------------------------------------------------------------------------------|
| `per-field`  | Default. One OME-TIFF per `FieldIndex` — equivalent to saving every tile as its own file.          |
| `stitch`     | Blend each `PartialTileIndex` group of tiles into a single mosaic OME-TIFF with feathered overlap. |

In `stitch` mode the output filename uses `M{partial_tile_index}` in place of
`F{field_index}`, e.g. `plate1_A01_M01_L1_A1_3D_2x.ome.tif`. Non-tiled records
remain `F{field_index}` regardless of `--tile-mode`.

### Multiple recordings of the same well: merge or split?

By default, when a directory contains multiple `.wpi` files that share the same
well/field/timeline/action (e.g. an experiment that imaged the same plate on
three consecutive days), their timepoints are concatenated along T into a single
OME-TIFF. That's the right thing for slow long-term timelapses.

For rapid time-lapses — typically `BF` or `2D` actions captured at ~20 Hz —
each WPI run is its own short burst that should not be glued onto its
neighbours. Use `--no-merge-actions` to split them:

```bash
compile-cv8000 --root-dir /data/plate --out-dir /out --no-merge-actions BF,2D
```

For matching actions, each `acquisition_index` gets its own OME-TIFF with an
`_R{idx:02d}` suffix (R = "recording"), e.g. `plate_A01_F01_L1_A1_BF_60x_R02.ome.tif`.
Other actions still merge as before.

### OME metadata

Each output file embeds the following in the `ImageDescription` (OME-XML):

- Standard OME fields: `PhysicalSizeX/Y` (µm), `Channels`, `TimeIncrement` (s).
- A JSON blob in `<Description>` with `Timestamps` (per-timepoint absolute ISO
  strings), `RelativeTimes` (seconds since first frame), `FrameIntervals`,
  `FramerateHz` (1 / median Δt), and the configured `TimelinePeriod` /
  `TimelineInterval` / `TimelineExpectedTime` from the `.mes` file. This also
  carries `WellID`, `FieldIndex` or `PartialTileIndex`, `ActionIndex`, `Action`,
  `ZMode`, plus `AcquisitionIndex` / `MergedAcquisitions` when split.

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

### Running long jobs over SSH / VS Code Remote

Compilation runs can take hours. To keep the job alive after closing VS Code or disconnecting SSH, detach it from the terminal:

```bash
# fire and forget — log to out.log, print PID
nohup uv run compile-cv8000 --root-dir /path/to/data --out-dir /path/to/output > out.log 2>&1 &

# follow progress
tail -f out.log

# stop it later
kill <PID>
```

Or use `tmux` if you want to reattach and watch live:

```bash
tmux new -s compile
uv run compile-cv8000 --root-dir /path/to/data --out-dir /path/to/output
# Ctrl-b then d to detach
tmux attach -t compile   # reconnect later
```

Notes:
- `uv run` resolves the project's virtualenv automatically — no need to activate `.venv` first.
- Run from the project root so `uv` finds `pyproject.toml`.
- VS Code's integrated terminal kills child processes when the window closes; `nohup` (or `tmux`) is what actually detaches them.

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

### OME-Zarr output

In addition to OME-TIFF, the compiler can write **OME-NGFF v0.4** (`.ome.zarr`)
datasets via `--format`:

```bash
# OME-Zarr only
compile-cv8000 --root-dir /data/plate --out-dir /out --format zarr

# both OME-TIFF and OME-Zarr
compile-cv8000 --root-dir /data/plate --out-dir /out --format both
```

`zarr` and `both` require the optional `[zarr]` extra (see [Installation](#installation)).
Each dataset is a 4-level multiscale pyramid (YX decimation, dtype preserved),
Blosc/zstd compressed. The non-standard CV8000 fields that OME-TIFF stores as a
JSON blob in `<Description>` live here under the `cv8000` namespace in `.zattrs`,
and per-channel display windows are written to `omero.channels`.

### Desktop GUI

A Tkinter front-end over the same pipeline, for users who prefer not to use the
command line:

```bash
uv run python gui/app.py
```

It exposes the same options as the CLI — source/output folders, fluorescence and
brightfield z-modes, tile mode, output format, no-merge actions, exclude keyword,
worker count and overwrite — and streams the compiler log and a progress bar into
the window while a run proceeds on a background thread.

To ship it as a standalone application (no Python install needed on the target
machine), build a one-folder bundle with PyInstaller:

```bash
bash gui/build.sh
# → dist/cv8000_compiler/cv8000_compiler
```

![CV8000 Compiler GUI](gui/screenshot.png)

### OME-Zarr explorer

`experiments/build_zarr_explorer.py` generates a single-page `index.html` for an
output directory of `.ome.zarr` datasets: an auto-detected plate map (96 / 384 /
1536-well), a dataset list filterable by well and action, and an embedded
[Vizarr](https://github.com/hms-dbmi/vizarr) viewer for the selected dataset.

```bash
# 1. build the page
uv run python experiments/build_zarr_explorer.py /path/to/out_dir

# 2. serve it (Vizarr fetches chunks over HTTP — file:// will not work)
python -m http.server 8000 --directory /path/to/out_dir
# then open http://localhost:8000/
```

If a `{stem}.analysis.html` sits next to a `{stem}.ome.zarr`, it is shown in a
second pane under the viewer when that dataset is selected.

### MP4 previews

`experiments/build_mp4_previews.py` turns compiled `*.ome.tif` timelapses into
small 8-bit H.264 MP4s for quick browsing. Each frame is percentile-normalized to
8-bit and 2× downscaled, shrinking ~1 TB of raw fluorescence timelapses to a
~12 GiB archive (~14× smaller in practice). Requires the `[video]` extra (see
[Installation](#installation)).

```bash
uv run python experiments/build_mp4_previews.py /path/to/out_dir
# writes MP4s to <out_dir>/mp4/, skipping brightfield (BF) acquisitions
```

Brightfield acquisitions are skipped by default (`--exclude BF`); tune quality and
size with `--crf`, `--scale`, and `--fps`. The run is parallel (`--workers`) and
idempotent (existing outputs are skipped). **The MP4s are lossy viewing previews
only** — 8-bit, downscaled, per-file contrast stretch — so keep the raw
`.ome.tif` as the quantitative source.

## Python API

The package can also be used programmatically:

```python
from pathlib import Path
from compiler.discovery import find_measurements
from compiler.processing import parse_measurements
from compiler.export import process_fieldstacks_parallel

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

## TIFF Compression Utility

A CLI tool and Python API for batch-compressing TIFF files. Accepts any mix of `.zip` archives, directories (scanned recursively), and individual `.tif`/`.tiff` files. Non-TIFF files are copied through unchanged.

Recompresses TIFFs with `zlib`, `lzw`, `lzma`, or `zstd` codecs while preserving ImageJ metadata, resolution tags, and photometric interpretation. Each output file is verified after compression. Uses `ProcessPoolExecutor` for true parallelism (bypasses the GIL).

### CLI usage

```bash
python -m tiff_utils.tiff_compression [OPTIONS] INPUTS...
```

```text
positional arguments:
  inputs                Zip files, directories, or individual files to process

options:
  -o, --output-dir DIR  Output directory (default: ./compressed_tiffs)
  -c, --compression     Codec: zlib | lzw | lzma | zstd (default: zlib)
  -j, --workers N       Parallel worker processes (default: 4)
  -Z, --max-parallel-zips N
                        Zip files to process concurrently (default: 1)
  --delete              Delete original zip files after successful verification
  --dry-run             Preview mode — process without deleting originals
```

### Compression examples

```bash
# Compress all TIFFs in a zip archive
python -m tiff_utils.tiff_compression data.zip -o compressed/

# Compress a whole directory with zstd, 8 workers
python -m tiff_utils.tiff_compression /data/experiment1 -c zstd -j 8 -o compressed/

# Mix of zips and loose files, delete originals after success
python -m tiff_utils.tiff_compression plate1.zip plate2.zip extra.tif --delete -o compressed/
```

### Python API (compression)

```python
from pathlib import Path
from tiff_utils.tiff_compression import compress_tiffs

summary = compress_tiffs(
    inputs=[Path("data.zip"), Path("/data/loose_tiffs/")],
    output_dir=Path("compressed"),
    compression="zstd",
    workers=8,
    delete_originals=False,
)
# summary: {"processed": N, "succeeded": N, "failed": N, "deleted": N}
```

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

- **`src/compiler/`** — CLI tool for compiling CV8000 image data
  - `cli.py` — CLI entry point (Typer)
  - `discovery.py` — recursive `.wpi` file discovery
  - `metadata.py` — Pydantic models for Yokogawa XML metadata
  - `processing.py` — XML parsing and DataFrame merging
  - `export.py` — OME-TIFF writing, corrections, and Z-projection
  - `export_zarr.py` — OME-Zarr (OME-NGFF v0.4) writer
- **`gui/`** — Tkinter desktop front-end for the compiler
  - `app.py` — the GUI application
  - `build.sh` / `cv8000_compiler.spec` — PyInstaller one-folder bundle
- **`experiments/`** — analysis and utility scripts
  - `build_zarr_explorer.py` — static HTML explorer for OME-Zarr output
  - `build_mp4_previews.py` — 8-bit H.264 MP4 previews from OME-TIFF timelapses
- **`src/tiff_utils/`** — batch TIFF compression utility
  - `tiff_compression.py` — compress TIFFs from zips, directories, or loose files
- **`experiment_manager/`** — Gradio web app for experiment registration
  - `app.py` — Gradio application
  - `Dockerfile` — container image
  - `nginx.conf` — reverse proxy config
- **`docker-compose.yml`** — orchestrates Gradio + nginx services

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.
