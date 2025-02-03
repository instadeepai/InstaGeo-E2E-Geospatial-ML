# InstaGeo - Data (Chip Creator)

## Overview
Working with raw satellite data typically demands a deep understanding of data processing techniques and specialized expertise, which can create significant barriers for those looking to utilize this data effectively, particularly in initiatives aimed at driving social good

InstaGeo's Chip Creator aims to fill this gap by enabling researchers to leverage multispectral imagery without the need to handle or process raw satellite data, making it easier to focus on analysis and application.

By providing gelocated observation records containing 'longitude', 'latitude', 'date' and 'label', chip creator sources the appropriate tiles and creates valid chips with corresponding segmentation maps.

## Data Sources
### Harmonized Landsat Sentinel - HLS
The [Harmonized Landsat-8 Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) satellite data product is en effort by [NASA](https://www.nasa.gov/) to provide a consistent and long-term data record of the Earth's land surface at a high temporal (2-3 days) and spatial (30m) resolutions.

### Sentinel-2
Sentinel-2, part of [ESA](https://www.esa.int/)'s [Copernicus Program](https://www.copernicus.eu/en), provides multispectral imagery across 13 bands with spatial resolutions of 10m, 20m, and 60m. It offers global coverage every 5 days, supporting applications like agriculture, land monitoring, and disaster response with free and open access to its data.

## Workflow
- Input geolocated observation records in a CSV file with the following columns:
    - x (longitude)
    - y (latitude)
    - date
    - label
- Group records based on Military Grid Reference System (MGRS)
- For each record, create a corresponding temporal series of granules
- Create a set of all granules
- Download the required bands from each granule
- Create and save chips and segmentation maps

Chip creator is particularly useful for geospatial image segmentation where large satellite images are segmented into smaller chips for training geospatial models (such as [Prithvi](https://huggingface.co/ibm-nasa-geospatial/Prithvi-100M)) using image segmentation objective.

## Requirements
- Python >= 3.10
- Required Python libraries listed in `requirements.txt`
- Ensure that wget is installed on your system

## Installation
Ensure that Python 3.10 and all required libraries are installed. You can install the required libraries using pip:

```bash
pip install -r requirements.txt
```
## Authentication
### Harmonized Landsat Sentinel - HLS
HLS data is hosted on **LP DAAC** and requires authentication to access it. Create an account an [Earth Data Account](https://urs.earthdata.nasa.gov/) and save the following to your home directory at `~/.netrc`

```plaintext
 machine urs.earthdata.nasa.gov login USERNAME password PASSWORD
```
 Replace `USERNAME` and `PASSWORD` with those of your account.
### Sentinel-2
Sentinel-2 data is hosted on the **Copernicus Data Space Ecosystem**, which requires authentication for access. Follow these steps to set up your credentials:

 1. Create an Account
Register for an account on the **Copernicus Data Space Ecosystem**:
[Copernicus Data Space Registration](https://dataspace.copernicus.eu/)

For detailed registration instructions, refer to the [Registration Documentation](https://documentation.dataspace.copernicus.eu/Registration.html).



2. Save Your Credentials
After registering, create a file called `.credentials` in your home directory (`~/.credentials`) with the following format:

```plaintext
USERNAME=your_email@example.com
PASSWORD=your_password
CLIENT_ID=cdse-public
```
Replace `your_email@example.com` and `your_password`  with those of your account.
The CLIENT_ID must remain cdse-public.

## Usage

### Command-line Arguments

- `--dataframe_path`: Path to the DataFrame CSV file. (required)
- `--chip_size`: Size of each chip. (default: 256)
- `--src_crs`: CRS of the geo-coordinates in dataframe_path. (default: 4326)
- `--output_directory`: Directory where the chips and segmentation maps will be saved. (required)
- `--min_count`: Minimum observation counts per tile, this is useful for controlling sparsity. (default: 100, minimum: 1)
- `--shift_to_month_start`: Indicates whether to shift the observation date to the beginning of the month. (default: True)
- `--is_time_series_task`: Indicates whether the task is a time series one. If True, data will be retrieved before the observation date. (default: True)
- `--num_steps`: Number of temporal steps. If is_time_series_task is True, an attempt will be made to retrieve *num_steps* chips prior to the observation date. (default: 3, minimum: 1)
- `--temporal_step`: Temporal step size in days. Used when fetching time-series data up to temporal_step days before the observation date. (default: 30)
- `--temporal_tolerance`: Tolerance (in days) used when searching for the closest tile. (default: 5)
- `--data_source`: Data source to use. Options: HLS, S2. (default: HLS)
- `--cloud_coverage`: Percentage of cloud cover to use. Accepted values: 0-100. (default: 10)
- `--window_size`: Defines the size of the window around the observation pixel. A value of 1 creates a 3×3 pixel window centered on the observation. (default: 0, minimum: 0)
- `--processing_method`: Method to use for processing tiles.(default: cog) Options:

    \- **"cog"**: Uses Cloud Optimized GeoTIFFs (COGs) to create chips.

    \- **"download"**: Downloads entire tiles for chip creation.

    \- **"download-only"**: Downloads tiles without further processing.

- `--mask_types`: List of different types of masking to apply. (default: [])
- `--masking_strategy`: Strategy for applying masking.(default: each) Options:

    \- **"each"**: Timestep-wise masking.

    \- **"any"**: Excludes pixels if the mask is present in at least one timestep.





### Running the Module
To run the module, use the following command in the terminal:

```bash
python chip_creator.py --dataframe_path="<path_to_dataframe_csv>" --chip_size=<chip_size> --output_directory="<output_directory>" --no_data_value=<no_data_value> --src_crs=<source_crs>
```

Replace `<hls_tile_path>`, `<path_to_dataframe_csv>`, `<chip_size>`, `<output_directory>`, `<no_data_value>` `<source_crs>` and `<destination_crs>`  with appropriate values.

### Example with HLS
```bash
python chip_creator.py --dataframe_path="/path/to/data.csv" --chip_size=224 --output_directory="/path/to/output" --no_data_value=-1 --src_crs 4326
```

This command will process the tile, segment it into chips of size 224x224, create segmentation maps based on the provided DataFrame, and save the output in the specified directory.

Run this command to test using an example observation record.

```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path=tests/data/test_breeding_data.csv \
    --output_directory="." \
    --min_count=4 \
    --chip_size=512 \
    --no_data_value=-1 \
    --temporal_tolerance=3 \
    --temporal_step=30 \
    --num_steps=3
```

### Example with S2
Run this command to achieve the same through Sentinel-2.
```bash
python -m "instageo.data.chip_creator" \
     --dataframe_path=tests/data/test_breeding_data.csv \
     --output_directory="." \
     --min_count=4 \
     --chip_size=512 \
     --no_data_value=-1 \
     --temporal_tolerance=3 \
     --temporal_step=30 \
     --num_steps=3 \
     --data_source=S2
```
