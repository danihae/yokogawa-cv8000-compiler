# Yokogawa Microscopy Utilities

A set of tools for the UMG Pharmacology microscopy workflow:

1. **CV8000 Data Compiler** — CLI tool that reads Yokogawa CV8000 output, compiles multi-channel / multi-timepoint / multi-Z data into TIFF stacks, and applies z-projections
2. **Microscopy Experiment Manager** — Gradio web app for registering experiments, creating metadata-tagged folders on the NAS, and managing multi-day acquisitions

## CV8000 Data Compiler

### Features

- Parses Yokogawa CV8000 MeasurementData.mlf files to extract metadata
- Reads associated MeasurementDetail.mrf files for additional metadata
- Compiles multi-channel, multi-timepoint, and multi-Z data into organized image stacks
- Creates maximum intensity projections (MIP), maximum average projections (MAP), or maximum entropy slice projections (MES)
- Preserves important metadata (timestamps, conditions, pixel size, etc.)
- Supports parallel processing for faster compilation

### Installation

Clone this repository and install:

```bash
git clone https://github.com/danihae/yokogawa-cv8000-compiler.git
cd yokogawa-cv8000-compiler
pip install -e .
```

### Usage

```bash
compile-cv8000 /path/to/data /path/to/output
```

Or as a Python module:

```bash
python -m yokogawa_cv8000_compiler /path/to/data /path/to/output
```

### Command-line options

```text
positional arguments:
  data_folder           Path to folder containing raw Yokogawa CV8000 data
  export_folder         Path to folder for compiled output

optional arguments:
  --depth DEPTH         Subdirectory depth to search for MeasurementData.mlf (default: 3)
  -p, --processes N     Number of parallel processes (default: 4)
  --proj-mode {mip,map,mes}
                        Projection mode (default: map):
                          mip  Maximum Intensity Projection — pixel-wise max across all z-slices
                          map  Maximum Average slice — z-slice with the highest mean intensity
                          mes  Maximum Entropy Slice — z-slice with the highest Shannon entropy (useful for focus-based selection)
```

### Examples

```bash
# Basic usage
compile-cv8000 /data/experiment1 /output/compiled

# Use 8 parallel processes with maximum intensity projection
compile-cv8000 /data/experiment1 /output/compiled -p 8 --proj-mode mip

# Use maximum entropy slice selection (good for in-focus plane detection)
compile-cv8000 /data/experiment1 /output/compiled --proj-mode mes

# Search only 2 subdirectory levels deep
compile-cv8000 /data/experiment1 /output/compiled --depth 2
```

### Output

The compiled data is saved as TIFF stacks (.tif) with ImageJ-compatible metadata (pixel size, axes).

Each file is named using this pattern:

```text
plate{plate}_well{well}_field{field}_channel{channel}_color{color}_action{action}.tif
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

## Dependencies

### CLI tool

- numpy, pandas, tifffile, xmltodict

### Web GUI

- gradio, python-dotenv

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## TODO

- [ ] Add a small real CV8000 measurement dataset (minimal plate, single well/field) to `tests/data/` for integration testing
- [ ] Add support for additional microscopy softwares (Yokogawa CQ1)