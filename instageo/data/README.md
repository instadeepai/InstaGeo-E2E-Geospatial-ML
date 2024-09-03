# InstaGeo - Data (Chip Creator)

## Overview
The [Harmonized Landsat-8 Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) satellite data product is en effort by [NASA](https://www.nasa.gov/) to provide a consistent and long-term data record of the Earth's land surface at a high temporal (2-3 days) and spatial (30m) resolutions.

As with every raw satellite data product, processing and manipulating it requires quite some knowledge and experience which can be an impediment for the use of such data for social good.

InstaGeo's Chip Creator aims to fill this gap to enable researchers leverage HLS without having to worry about handling and processing raw satellite data.

By providing gelocated observation records containing longitude, latitude, date and label, chip creator sources the appropriate HLS granules and creates valid chips with corresponding segmentation maps.

## Workflow
- Input geolocated observation records in a CSV file with the following columns:
    - x (longitude)
    - y (latitude)
    - date
    - label
- Group records based on Military Grid Reference System (MGRS)
- For each record, create a corresponding temporal series of HLS granules
- Create a set of all HLS granules
- Download the required bands from each granule
- Create and save chips and segementation maps

Chip creator is particularly useful for geospatial image segmentation where large satellite images are segmented into smaller chips for training geospatial models (such as [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)) using image segmentation objective.

## Requirements
- Python >= 3.10
- Required Python libraries listed in `requirements.txt`

## Installation
Ensure that Python 3.10 and all required libraries are installed. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```
## Authentication
HLS data is hosted on LP DAAC and requires authentication to access it. Create an account an [Earth Data Account](https://urs.earthdata.nasa.gov/) and save the following to your home directory at `~/.netrc`

 `machine urs.earthdata.nasa.gov login USERNAME password PASSWORD`

 Replace `USERNAME` and `PASSWORD` with those of your account.

## Usage

### Command-line Arguments
- `--dataframe_path`: Path to the DataFrame CSV file containing geolocated observation records. (required)
- `--chip_size`: Size of each chip. (default: 224)
- `--output_directory`: Directory where the chips and segmentation maps will be saved. (required)
- `--no_data_value`: Value to use for no data pixels in the segmentation maps. (default: -1)
- `--src_crs`: Coordinate Reference System (CRS) of the geographical coordinates in `dataframe_path`. (default: 4326)
- `--min_count`: Minimum records that is desired per granule. This is useful for cotroling sparsity. (default: 1)
- `--num_steps`: Number of temporal steps for creating data. For static data, this value should be set to 1 (default: 1)
- `--temporal_step`: Size of each temporal step in days (default: 30)
- `--temporal_tolerance`: Number of days to be tolerated when searching for closest HLS granules. (default: 3)
- `--download_only`: Downloads only HLS tiles when set to True. (default: False)


### Running the Module
To run the module, use the following command in the terminal:

```bash
python chip_creator.py --dataframe_path="<path_to_dataframe_csv>" --chip_size=<chip_size> --output_directory="<output_directory>" --no_data_value=<no_data_value> --src_crs=<source_crs>
```

Replace `<hls_tile_path>`, `<path_to_dataframe_csv>`, `<chip_size>`, `<output_directory>`, `<no_data_value>` `<source_crs>` and `<destination_crs>`  with appropriate values.

### Example
```bash
python chip_creator.py --dataframe_path="/path/to/data.csv" --chip_size=224 --output_directory="/path/to/output" --no_data_value=-1 --src_crs 4326
```

This command will process the HLS tile, segment it into chips of size 224x224, create segmentation maps based on the provided DataFrame, and save the output in the specified directory.

Run this command to test using an example observation record.

```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path=tests/data/test_breeding_data.csv
    --output_directory="." \
    --min_count=4 \
    --chip_size=512 \
    --no_data_value=-1 \
    --temporal_tolerance=3 \
    --temporal_step=30 \
    --num_steps=3
```
