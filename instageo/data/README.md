# InstaGeo Chip Creator

## Overview
The InstaGeo Chip Creator is a Python module designed for creating chips and corresponding segmentation maps from [Harmonized Landsat and Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) Data Product. This tool is particularly useful for geospatial image segmentation where large satellite images are segmented into smaller chips for training geospatial models (such as [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)) using image segmentation objective.

## Requirements
- Python >= 3.10
- Required Python libraries listed in `requirements.txt`

## Installation
Ensure that Python 3.10 and all required libraries are installed. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```

## Usage

### Command-line Arguments
- `--hls_tile_path`: Path to HLS tile directory. (required)
- `--dataframe_path`: Path to the DataFrame CSV file containing the data for segmentation maps. (required)
- `--chip_size`: Size of each chip. (default: 224)
- `--output_directory`: Directory where the chips and segmentation maps will be saved. (required)
- `--no_data_value`: Value to use for no data areas in the segmentation maps. (default: -1)
- `--src_crs`: Coordinate Reference System (CRS) of the geographical coordinates in `dataframe_path`. (default: 4326)
- `--dst_crs`: , CRS of the geographical coordinates in `hls_tile_path`. (default: 32613)

### Running the Module
To run the module, use the following command in the terminal:

```bash
python chip_creator.py --hls_tile_path="<path_to_hls_tile_directory>" --dataframe_path="<path_to_dataframe_csv>" --chip_size=<chip_size> --output_directory="<output_directory>" --no_data_value=<no_data_value> --src_crs=<source_crs> --dst_crs=<destination_crs>
```

Replace `<hls_tile_path>`, `<path_to_dataframe_csv>`, `<chip_size>`, `<output_directory>`, `<no_data_value>` `<source_crs>` and `<destination_crs>`  with appropriate values.

### Example
```bash
python instageo_chip_creator.py --geotiff_path="/path/to/geotiff.tif" --dataframe_path="/path/to/data.csv" --chip_size=224 --output_directory="/path/to/output" --no_data_value=-1 --src_crs 4326 --dst_crs 32613
```

This command will process the HLS tile, segment it into chips of size 224x224, create segmentation maps based on the provided DataFrame, and save the output in the specified directory.
