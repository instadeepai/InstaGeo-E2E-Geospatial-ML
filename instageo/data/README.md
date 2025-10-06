# InstaGeo - Data

## Overview
Working with raw satellite data typically demands a deep understanding of data processing techniques and specialized expertise, which can create significant barriers for those looking to utilize this data effectively, particularly in initiatives aimed at driving social good

InstaGeo aims to fill this gap by enabling researchers to leverage multispectral imagery without the need to handle or process raw satellite data, making it easier to focus on analysis and application.

## Data Sources
### Harmonized Landsat Sentinel - HLS
The [Harmonized Landsat-8 Sentinel-2 (HLS)](https://hls.gsfc.nasa.gov/) satellite data product is en effort by [NASA](https://www.nasa.gov/) to provide a consistent and long-term data record of the Earth's land surface at a high temporal (2-3 days) and spatial (30m) resolutions.

### Sentinel-2
Sentinel-2, part of [ESA](https://www.esa.int/)'s [Copernicus Program](https://www.copernicus.eu/en), provides multispectral imagery across 13 bands with spatial resolutions of 10m, 20m, and 60m. It offers global coverage every 5 days, supporting applications like agriculture, land monitoring, and disaster response with free and open access to its data.

### Sentinel-1
Sentinel-1, part of [ESA](https://www.esa.int/)'s [Copernicus Program](https://www.copernicus.eu/en), provides all-weather, day-and-night radar imagery using C-band Synthetic Aperture Radar (SAR). It offers global coverage with revisit times of 6 to 12 days, supporting applications like land and ocean monitoring, emergency response, and environmental surveillance with free and open access to its data.

## Requirements
- Python >= 3.10
- Install dependencies using uv (recommended)
- Ensure that wget is installed on your system

## Installation
Ensure that Python 3.10 is installed. Install InstaGeo data dependencies using uv:

```bash
uv sync --locked --extra data --extra cpu
```
## Authentication
### Harmonized Landsat Sentinel - HLS
HLS data is hosted on **LP DAAC** and requires authentication to access it. Create an account an [Earth Data Account](https://urs.earthdata.nasa.gov/) and save the following to your home directory at `~/.netrc`

```plaintext
 machine urs.earthdata.nasa.gov login USERNAME password PASSWORD
```
 Replace `USERNAME` and `PASSWORD` with those of your account.
### Sentinel-2 and Sentinel-1
Sentinel data is hosted on the **Copernicus Data Space Ecosystem**, which requires authentication for access. Follow these steps to set up your credentials:

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

# Configuration Settings
InstaGeo uses Pydantic settings that can be overridden using environment variables. Here are the available settings and how to override them:

## GDAL Options
```bash
export GDALOPTIONS_CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.jpg"
export GDALOPTIONS_GDAL_CACHEMAX=2048  # 2GB instead of 1GB
export GDALOPTIONS_GDAL_HTTP_MAX_RETRY="20"
export GDALOPTIONS_GDAL_HTTP_RETRY_DELAY="1.0"
```

## No Data Values
```bash
export NODATAVALUES_HLS=255
export NODATAVALUES_S2=255
export NODATAVALUES_S1=0
export NODATAVALUES_SEG_MAP=0
```

## HLS Block Sizes
```bash
export HLSBLOCKSIZES_X=512
export HLSBLOCKSIZES_Y=512
```

## HLS API Settings
```bash
export HLSAPISETTINGS_URL="https://your-custom-url.com"
export HLSAPISETTINGS_COLLECTIONS='["HLSL30_2.0"]'
```

## Data Pipeline Settings
```bash
export DATAPIPELINESETTINGS_BATCH_SIZE=32
export DATAPIPELINESETTINGS_METADATA_SEARCH_RATELIMIT=20
export DATAPIPELINESETTINGS_COG_DOWNLOAD_RATELIMIT=40
```

You can also create a `.env` file in your project root to define these variables. The settings will be loaded in this order of precedence:
1. Environment variables
2. `.env` file
3. Default values in the code

# Chip Creator
The Chip Creator script is designed for scenarios where label data is provided in a CSV file format. It automates the extraction of relevant satellite tiles and the generation of image chips with corresponding segmentation maps. By using geolocated observation records that include longitude, latitude, date, and label, the script sources the appropriate satellite tiles and creates valid chips, each paired with an accurate segmentation map.

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
- `--data_source`: Data source to use. Options: HLS, S2 and S1. (default: HLS)
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
python chip_creator.py --dataframe_path="<path_to_dataframe_csv>" --chip_size=<chip_size> --output_directory="<output_directory>"  --src_crs=<source_crs>
```

Replace `<tile_path>`, `<path_to_dataframe_csv>`, `<chip_size>`, `<output_directory>`, `<source_crs>` and `<destination_crs>`  with appropriate values.

### Example with HLS
```bash
python chip_creator.py --dataframe_path="/path/to/data.csv" --chip_size=224 --output_directory="/path/to/output"  --src_crs=4326
```

This command will process the tile, segment it into chips of size 224x224, create segmentation maps based on the provided DataFrame, and save the output in the specified directory.

Run this command to test using an example observation record.

```bash
python -m "instageo.data.chip_creator" \
    --dataframe_path=tests/data/test_breeding_data.csv \
    --output_directory="." \
    --min_count=4 \
    --chip_size=512 \
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
     --temporal_tolerance=7 \
     --temporal_step=30 \
     --num_steps=3 \
     --data_source=S2
```

### Example with S1
Run this command to achieve the same through Sentinel-1.
```bash
python -m "instageo.data.chip_creator" \
     --dataframe_path=tests/data/test_breeding_data.csv \
     --output_directory="." \
     --min_count=4 \
     --chip_size=512 \
     --temporal_tolerance=13 \
     --temporal_step=30 \
     --num_steps=3 \
     --data_source=S1
```





# Raster Chip Creator
Raster Chip Creator is a script designed for scenarios where no CSV file containing labels or metadata is provided. Instead, all necessary pieces of information are extracted directly from raster files or a directory containing multiple rasters. These raster files encode target values as pixel-level data, where each pixel represents a specific label or class tied to its geographic location. Acting as built-in segmentation maps, the rasters guide the generation of satellite image chips and corresponding segmentation masks. The script automates the entire process, from tile extraction to chip creation, based on a chip size parameter provided by the user.

It is ideal for situations where you have geospatial data in raster format, and you want to generate image chips for further analysis or training machine learning models.

## Workflow
- Input Data:
Provide a folder containing one or more raster files. Each raster will contain geospatial information, and the tool will use the raster's pixel values to generate segmentation maps.

- Extract Tiles:
The tool will automatically extract tiles from the provided raster(s) based on the geolocation information encoded in the raster file(s).

- Crop Tiles to Chips:
For each tile, the tool will crop it into smaller, fixed-size chips (e.g., 256x256 pixels or another defined size). Each chip will be created from a specific part of the raster.

- Generate Segmentation Maps:
Each chip will be paired with a corresponding segmentation map, which is derived from the pixel values of the raster.

- Save Chips and Segmentation Maps:
The cropped chips and their segmentation maps will be saved to a specified output directory, ready for use in downstream tasks like analysis or model training.

## Usage

### Command-line Arguments
- `--raster_path`: Path to a single raster file or a folder containing multiple raster files. These files will be used to extract the tiles and generate the chips.

- `--records_file`: Path to input records file containing at least date and geometry columns.

- `--src_crs`: Coordinate Reference System (CRS) for the input raster data.

- `--spatial_resolution`: Defines the ground sampling distance (in units of the specified CRS) for both the output chips and segmentation maps. Both outputs will use the same CRS specified by `src_crs` and will have the specified resolution. For the raster chip creator, the input segmentation map must have this resolution. For the regular chip creator, the output segmentation map will be generated at this resolution.

- `--chip_size`: Size of each chip in pixels.

- `--output_directory`: Directory where the generated chips and segmentation maps will be saved.

- `--num_steps`: Defines the number of temporal steps to retrieve from the raster(s). For time series data, this will retrieve chips from num_steps prior to the observation date.

- `--temporal_step`: Temporal step size (in days) for fetching data in time-series tasks.

- `--temporal_tolerance`: Tolerance in days for finding the closest tile in temporal data.

- `--data_source`: Select the satellite data source (e.g., HLS, S2, S1).

- `--cloud_coverage`: Defines the maximum allowed cloud coverage for tiles. Valid values are between 0 and 100.

- `--mask_types`: List which quality masks to apply using the HLS QA bitmask. These masks help exclude unwanted regions (like clouds or water) from your chips (cloud, near_cloud_or_shadow, cloud_shadow and water). See section 6.4 of [HLS User Guide](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf) for more details on the mask types.

- `--masking_strategy`: Defines the masking strategy:

    \- **"each"**: Masking applied for each individual timestep.

    \- **"any"**: Masking applied if any timestep contains a mask.

- `--qa_check`: Enables quality control checks to filter out invalid chips. Removes chips that have no valid data after cloud masking, as well as segmentation maps that contain no valid labels after masking.

- `--daytime_only`: Filters out nighttime observations. Using the `datetime` and `bbox` fields of a STAC item, it checks if the satellite data was recorded between sunrise and sunset.


### Running the Module
To run the module, use the following command in the terminal:

```bash
python raster_chip_creator.py \
 --raster_path /path/to/raster.tif \
 --output_dir /path/to/output \
 --chip_size 256
```

## Advanced Usage

### Multi-Source Data Retrieval
InstaGeo efficiently searches for and downloads multi-spectral earth observation images from:
- **HLS (Harmonized Landsat Sentinel-2)**: 30m resolution, 2-3 day temporal coverage
- **Sentinel-2**: 10-60m resolution, 5-day revisit time, 13 spectral bands
- **Sentinel-1**: C-band SAR imagery for all-weather monitoring

### Advanced Data Processing Features
- **Automated Chip Creation**: Break down large satellite tiles into ML-ready patches with configurable sizes
- **Segmentation Map Generation**: Pixel-level categorization for training targets
- **Quality Assessment**: Built-in cloud masking, data validation, and filtering
- **Temporal Alignment**: Multi-temporal data stacking with configurable temporal tolerance

### Scalable Pipeline Architecture
- **Dask Integration**: Distributed processing for large-scale data operations
- **Configurable Settings**: Environment-based configuration for GDAL options, API endpoints, and processing parameters
- **Authentication Management**: Seamless integration with NASA EarthData and Copernicus Data Space Ecosystem

### Data Cleaning and Validation
Advanced data cleaning capabilities with customizable strategies for handling no-data values, cloud coverage, and observation quality.

## Best Practices

### Performance Optimization
- Use appropriate `batch_size` for your system's memory capacity
- Configure `temporal_tolerance` based on your data requirements
- Leverage Dask for distributed processing of large datasets

### Quality Control
- Always validate downloaded data before processing
- Use appropriate masking strategies for your use case
- Monitor cloud coverage thresholds for optical imagery

### Storage Management
- Organize output directories by date and data source
- Implement cleanup strategies for temporary files
- Consider compression for long-term storage of processed chips
