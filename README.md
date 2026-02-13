# Yokogawa CV8000 Data Compiler

A Python tool for processing output data from Yokogawa Cell Voyager CV8000 high-content screening systems. This tool reads the metadata files, compiles the data, and builds image stacks for each Field of View (FoV). This makes downstream data processing easier and eliminates the need for the proprietary Yokogawa CellProfiler tool.

## Features

- Parses Yokogawa CV8000 MeasurementData.mlf files to extract metadata
- Reads associated MeasurementDetail.mrf files for additional metadata
- Compiles multi-channel, multi-timepoint, and multi-Z data into organized image stacks
- Creates maximum intensity projections (MIP) or maximum average projections (MAP)
- Preserves important metadata (timestamps, conditions, pixel size, etc.)
- Supports parallel processing for faster compilation

## Installation

Clone this repository to your local machine:

```bash
git clone https://github.com/danihae/yokogawa-cv8000-compiler.git
cd yokogawa-cv8000-compiler
```

Install the package using pip:

```bash
pip install -e .
```

## Usage

Run the compiler from the command line:

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
  --proj-mode {mip,map} Projection mode: 'mip' (max intensity) or 'map' (max average) (default: map)
```

### Examples

```bash
# Basic usage
compile-cv8000 /data/experiment1 /output/compiled

# Use 8 parallel processes with maximum intensity projection
compile-cv8000 /data/experiment1 /output/compiled -p 8 --proj-mode mip

# Search only 2 subdirectory levels deep
compile-cv8000 /data/experiment1 /output/compiled --depth 2
```

The script will:

- Find all MeasurementData.mlf files in the specified directory
- Extract metadata from these files and associated MeasurementDetail.mrf files
- Process each field in each well of each plate
- Save compiled TIFF stacks to the export folder

### Output

The compiled data is saved as TIFF stacks (.tif) with ImageJ-compatible metadata (pixel size, axes).

Each file is named using this pattern:

```text
plate{plate}_well{well}_field{field}_channel{channel}_color{color}_action{action}.tif
```

## Code Structure

- **`__main__.py`**: CLI entry point that orchestrates the compilation process
- **`functions.py`**: Contains all utility functions for parsing and processing data

## Dependencies

- numpy: For numerical operations
- pandas: For data manipulation and management
- tifffile: For reading and writing TIFF files
- xmltodict: For parsing XML metadata files

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## TODO

- [ ] Add a small real CV8000 measurement dataset (minimal plate, single well/field) to `tests/data/` for integration testing
