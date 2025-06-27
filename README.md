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
git clone https://github.com/yourusername/yokogawa-cv8000-compiler.git
cd yokogawa-cv8000-compiler
```

Install the required dependencies using pip:

```bash
pip install -e .
```

## Usage

1. **Adjust the folder paths in `compile.py`**:

```python
# data folder - where your Yokogawa CV8000 data is stored
folder_data = 'X:/YourPath/YourData/'

# folder to save compiled data
folder_export = 'D:/YourPath/compiled/'
```

2. **Run the compilation script**:

```bash
python __main__.py
```

The script will:
- Find all MeasurementData.mlf files in the specified directory
- Extract metadata from these files and associated MeasurementDetail.mrf files
- Process each field in each well of each plate
- Save compiled TIFF stacks and CSV metadata files to the export folder

3. **Using the compiled data**:

The compiled data will be saved as:
- TIFF stacks (.tif) with ImageJ metadata
- CSV files with timestamps and condition information

Each file is named using this pattern:
```
plate{plate}_well{well}_field{field}_channel{channel}_color_{color}.tif
plate{plate}_well{well}_field{field}_channel{channel}_color_{color}.csv
```

4. **Customizing the compilation**:

You can modify the `compile_field` function in `functions.py` to change:
- Projection mode ('map' or 'mip')
- Output format
- Additional metadata to preserve

Example for adjusting the number of parallel processes:
```python
# In __main__.py
with Pool(4) as pool:  # Change from 2 to 4 for more parallel processes
    pool.starmap(compile_field_wrapper, args_list)
```

## Code Structure

- **compile.py**: Main script that orchestrates the compilation process
- **functions.py**: Contains all the utility functions for parsing and processing the data

## Dependencies

- numpy: For numerical operations
- pandas: For data manipulation and management
- tifffile: For reading and writing TIFF files
- xmltodict: For parsing XML metadata files

## License

This project is licensed under the MIT License - see the LICENSE file for details.
