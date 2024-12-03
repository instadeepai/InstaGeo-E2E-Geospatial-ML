# ------------------------------------------------------------------------------
# This code is licensed under the Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0) License.
#
# You are free to:
# - Share: Copy and redistribute the material in any medium or format
# - Adapt: Remix, transform, and build upon the material
#
# Under the following terms:
# - Attribution: You must give appropriate credit, provide a link to the license,
#   and indicate if changes were made. You may do so in any reasonable manner,
#   but not in any way that suggests the licensor endorses you or your use.
# - NonCommercial: You may not use the material for commercial purposes.
# - ShareAlike: If you remix, transform, or build upon the material, you must
#   distribute your contributions under the same license as the original.
#
# For more details, see https://creativecommons.org/licenses/by-nc-sa/4.0/
# ------------------------------------------------------------------------------

"""InstaGeo Chip Creator Module."""

import json
import os
from itertools import chain
from typing import Any

import dask.distributed
import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from absl import app, flags, logging
from dotenv import load_dotenv
from tqdm import tqdm

import instageo.data.hls_utils as hls_utils
from instageo.data.geo_utils import (
    MASK_DECODING_POS,
    apply_mask,
    open_hls_cogs,
    open_mf_tiff_dataset,
)
from instageo.data.settings import GDALOptions

load_dotenv(".credentials")

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame CSV file.")
flags.DEFINE_integer("chip_size", 256, "Size of each chip.")
flags.DEFINE_integer("src_crs", 4326, "CRS of the geo-coordinates in `dataframe_path`")
flags.DEFINE_string(
    "output_directory",
    None,
    "Directory where the chips and segmentation maps will be saved.",
)
flags.DEFINE_integer(
    "no_data_value", -9999, "Value to use for no data areas in the segmentation maps."
)
flags.DEFINE_integer(
    "min_count", 100, "Minimum observation counts per tile", lower_bound=1
)
flags.DEFINE_boolean(
    "shift_to_month_start",
    True,
    "Indicates whether or not to shift the observation date to the beginning of the month",
)
flags.DEFINE_boolean(
    "is_time_series_task",
    True,
    """Indicates whether or not the current task is a time series one. The data will be then
    retrieved before the date of observation""",
)
flags.DEFINE_integer(
    "num_steps",
    3,
    """Number of temporal steps. When `is_time_series_task` is set to True, an attempt
    will be made to retrieve `num_steps` HLS chips prior to the observation date.
    Otherwise, the value of `num_steps` will default to 1 and an attempt will be made to retrieve
    the HLS chip corresponding to the observation date.
    """,
    lower_bound=1,
)
flags.DEFINE_integer(
    "temporal_step",
    30,
    """Temporal step size. When dealing with a time series task, an attempt will be made to
    fetch the data up to `temporal_step` days away from the date of observation. A tolerance might
    be applied when fetching the data for the different time steps.""",
)
flags.DEFINE_integer(
    "temporal_tolerance", 5, "Tolerance used when searching for the closest tile"
)
flags.DEFINE_integer(
    "window_size",
    0,
    """Size of the window defined around the observation pixel. For instance, a value of 1 means
    that the label of the observation will be assigned to a 3x3 pixels window centered around the
    pixel of observation. The values are assigned within the bounds of a specific chip, i.e the
    window will be clipped to the extents of the chip in case it falls outside of the chip. A
    non-zero value for this parameter typically means that the observation covers more ground or
    pixels and can, in some cases account for low geolocation precision. Keep the default value
    if only interested in the pixel in which the observation falls.""",
    lower_bound=0,
)
flags.DEFINE_enum(
    "processing_method",
    "cog",
    ["cog", "download", "download-only"],
    """Method to use to process the tiles:
    - "cog" corresponds to creating the chips by utilizing the Cloud Optimized GeoTIFFs.
    - "download" corresponds to downloading entire HLS tiles to be used for creating
    the chips.
    - "download-only" corresponds to a simple download of the tiles without further
    processing.""",
)
flags.DEFINE_list(
    "mask_types",
    [],
    "List of different types of masking to apply",
)
flags.register_validator(
    "mask_types",
    lambda val_list: all(v in MASK_DECODING_POS.keys() for v in val_list),
    message=f"Valid values are {list(MASK_DECODING_POS.keys())}",
)
flags.DEFINE_enum(
    "masking_strategy",
    "each",
    ["each", "any"],
    """Method to use when applying masking:
    - "each" for timestep-wise masking.
    - "any" to exclude pixels if the mask is present for at least one timestep.
    """,
)
flags.DEFINE_integer(
    "cloud_coverage",
    10,
    "Percentage os cloud cover to use. Accepted values are between 0and 100.",
)


def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


def create_segmentation_map(
    chip: Any, df: pd.DataFrame, no_data_value: int, window_size: int
) -> xr.DataArray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        no_data_value (int): Value to be used for pixels with no data.
        window_size (int): Window size to use around the observation pixel.

    Returns:
         xr.DataArray: The created segmentation map as an xarray DataArray.
    """
    seg_map = xr.full_like(chip.isel(band=0), fill_value=no_data_value, dtype=np.int16)
    df = df[
        (chip["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= chip["x"].max().item())
        & (chip["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= chip["y"].max().item())
    ]
    cols, rows = np.floor(
        ~seg_map.rio.transform() * (df.geometry.x.values, df.geometry.y.values)
    ).astype(int)
    offsets = np.arange(-window_size, window_size + 1)
    offset_rows, offset_cols = np.meshgrid(offsets, offsets)
    window_rows = np.clip(
        rows[:, np.newaxis, np.newaxis] + offset_rows, 0, chip.sizes["x"] - 1
    )
    window_cols = np.clip(
        cols[:, np.newaxis, np.newaxis] + offset_cols, 0, chip.sizes["y"] - 1
    )
    window_labels = np.repeat(df.label.values, offset_rows.ravel().shape)
    seg_map.values[window_rows.ravel(), window_cols.ravel()] = window_labels
    return seg_map


def get_chip_coords(
    df: gpd.GeoDataFrame, tile: xr.DataArray, chip_size: int
) -> np.array:
    """Get Chip Coordinates.

    Given a list of x,y coordinates tuples of a point and an xarray dataarray, this
    function returns the unique corresponding x,y indices of the grid where each point will fall
    when the DataArray is gridded such that each grid has size `chip_size`
    indices where it will fall.

    Args:
        gdf (gpd.GeoDataFrame): GeoPandas dataframe containing the point.
        tile (xr.DataArray): Tile DataArray.
        chip_size (int): Size of each chip.

    Returns:
        List of chip indices.
    """
    cols, rows = np.floor(
        ~tile.rio.transform() * (df.geometry.x.values, df.geometry.y.values)
    ).astype(int)
    return np.unique(np.stack((cols // chip_size, rows // chip_size), axis=-1), axis=0)


def create_and_save_chips_with_seg_maps(
    processing_method: str,
    hls_tile_dict: dict[str, Any],
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_types: list[str],
    masking_strategy: str,
    window_size: int,
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a HLS tile and save them to
    an output directory.

    Args:
        processing_method (str): Processing method to use to create the chips and
        segmentation maps.
        hls_tile_dict (Dict): A dict mapping band names to HLS tile filepath.
        df (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be
            saved.
        no_data_value (int): Value to use for no data areas in the segmentation maps.
        src_crs (int): CRS of points in `df`
        mask_types (list[str]): Types of masking to perform.
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)
        window_size (int): Window size to use around the observation pixel.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    load_masks = True if mask_types else False
    dsb, dsm, crs = (
        open_hls_cogs(hls_tile_dict, load_masks=load_masks)
        if processing_method == "cog"
        else open_mf_tiff_dataset(hls_tile_dict, load_masks=load_masks)
    )
    df = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(x=df.x, y=df.y))
    df.set_crs(epsg=src_crs, inplace=True)
    df = df.to_crs(crs=crs)
    df = df[
        (dsb["x"].min().item() <= df["geometry"].x)
        & (df["geometry"].x <= dsb["x"].max().item())
        & (dsb["y"].min().item() <= df["geometry"].y)
        & (df["geometry"].y <= dsb["y"].max().item())
    ]
    os.makedirs(output_directory, exist_ok=True)
    tile_name_splits = hls_tile_dict["tiles"]["B02_0"].split(".")
    tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
    date_id = df.iloc[0]["date"].strftime("%Y%m%d")
    chips = []
    seg_maps: list[str | None] = []

    n_chips_x = dsb.sizes["x"] // chip_size
    n_chips_y = dsb.sizes["y"] // chip_size
    chip_coords = get_chip_coords(df, dsb, chip_size)
    for x, y in chip_coords:
        # TODO: handle potential partially out of bound chips
        if (x >= n_chips_x) or (y >= n_chips_y):
            continue
        chip_id = f"{date_id}_{tile_id}_{x}_{y}"
        chip_name = f"chip_{chip_id}.tif"
        seg_map_name = f"seg_map_{chip_id}.tif"

        chip_filename = os.path.join(output_directory, "chips", chip_name)
        seg_map_filename = os.path.join(output_directory, "seg_maps", seg_map_name)
        if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
            continue
        chip = dsb.isel(
            x=slice(x * chip_size, (x + 1) * chip_size),
            y=slice(y * chip_size, (y + 1) * chip_size),
        ).compute()
        chip = chip if processing_method == "cog" else chip.band_data
        if dsm is not None:
            chip_mask = dsm.isel(
                x=slice(x * chip_size, (x + 1) * chip_size),
                y=slice(y * chip_size, (y + 1) * chip_size),
            ).compute()
            chip_mask = chip_mask if processing_method == "cog" else chip_mask.band_data
            chip = apply_mask(
                chip=chip,
                mask=chip_mask,
                no_data_value=no_data_value,
                mask_types=mask_types,
                masking_strategy=masking_strategy,
            )
        if chip.where(chip != no_data_value).count().values == 0:
            continue
        seg_map = create_segmentation_map(chip, df, no_data_value, window_size)
        if seg_map.where(seg_map != no_data_value).count().values == 0:
            continue
        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chips.append(chip_name)
        chip.rio.to_raster(chip_filename)
    return chips, seg_maps


def create_hls_dataset(
    data_with_tiles: pd.DataFrame, outdir: str
) -> tuple[dict[str, dict[str, Any]], set[str]]:
    """Creates HLS Dataset.

    A HLS dataset is a list of dictionary mapping band names to corresponding GeoTiff
    filepath. It is required for creating chips.

    Args:
        data_with_tiles (pd.DataFrame): A dataframe containing observations that fall
            within a dense tile. It also has `hls_tiles` column that contains a temporal
            series of HLS granules.
        outdir (str): Output directory where tiles could be downloaded to.

    Returns:
        A tuple containing HLS dataset and a list of tiles that needs to be downloaded.
    """
    data_with_tiles = data_with_tiles.drop_duplicates(subset=["hls_tiles"])
    data_with_tiles = data_with_tiles[
        data_with_tiles["hls_tiles"].apply(
            lambda granule_lst: all("HLS" in str(item) for item in granule_lst)
        )
    ]
    assert not data_with_tiles.empty, "No observation record with valid HLS tiles"
    hls_dataset = {}
    data_links = []
    s30_bands = ["B02", "B03", "B04", "B8A", "B11", "B12", "Fmask"]
    l30_bands = ["B02", "B03", "B04", "B05", "B06", "B07", "Fmask"]

    for hls_tiles, download_links, obsv_date in zip(
        data_with_tiles["hls_tiles"],
        data_with_tiles["data_links"],
        data_with_tiles["date"],
    ):
        bands_paths = {}
        masks_paths = {}
        obsv_data_links = []
        for idx, (tile, tile_download_links) in enumerate(
            zip(hls_tiles, download_links)
        ):
            tile = tile.strip(".")
            bands_of_interest = s30_bands if "HLS.S30" in tile else l30_bands
            filtered_downloads_links = [
                next(link for link in tile_download_links if band + ".tif" in link)
                for band in bands_of_interest
            ]
            assert len(set(filtered_downloads_links)) == len(bands_of_interest)
            bands_paths.update(
                {
                    f"{band}_{idx}": os.path.join(
                        outdir, "hls_tiles", f"{tile}.{band}.tif"
                    )
                    for band in bands_of_interest[:-1]
                }
            )
            masks_paths.update(
                {
                    f"{bands_of_interest[-1]}_{idx}": os.path.join(
                        outdir, "hls_tiles", f"{tile}.{bands_of_interest[-1]}.tif"
                    )
                }
            )
            obsv_data_links.append(filtered_downloads_links)
        data_links.extend(obsv_data_links)
        hls_dataset[f'{obsv_date.strftime("%Y-%m-%d")}_{tile.split(".")[2]}'] = {
            "tiles": bands_paths,
            "fmasks": masks_paths,
            "data_links": obsv_data_links,
        }

    return hls_dataset, set(chain.from_iterable(data_links))


def setup() -> None:
    """Setup for environment to be used by Dask client.

    Configures relevant GDAL options for reading COGs
    """
    earthaccess.login(persist=True)
    env = rasterio.Env(**GDALOptions().model_dump())
    env.__enter__()


def main(argv: Any) -> None:
    """CSV Chip Creator.

    Given a csv file containing geo-located point observations and labels, the Chip
    Creator creates small chips from large HLS tiles which is suitable for training
    segmentation models.
    """
    del argv
    data = pd.read_csv(FLAGS.dataframe_path)
    data["date"] = (
        pd.to_datetime(data["date"]) - pd.offsets.MonthBegin(1)
        if FLAGS.shift_to_month_start
        else pd.to_datetime(data["date"])
    )
    data["input_features_date"] = (
        data["date"] - pd.DateOffset(days=FLAGS.temporal_step)
        if FLAGS.is_time_series_task
        else data["date"]
    )
    FLAGS.num_steps = 1 if not FLAGS.is_time_series_task else FLAGS.num_steps
    sub_data = hls_utils.get_hls_tiles(data, min_count=FLAGS.min_count)

    if not (
        os.path.exists(os.path.join(FLAGS.output_directory, "hls_dataset.json"))
        and os.path.exists(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )
    ):
        logging.info("Creating HLS dataset JSON.")
        logging.info("Retrieving HLS tile ID for each observation.")
        sub_data_with_tiles = hls_utils.add_hls_granules(
            sub_data,
            num_steps=FLAGS.num_steps,
            temporal_step=FLAGS.temporal_step,
            temporal_tolerance=FLAGS.temporal_tolerance,
        )
        logging.info("Retrieving HLS tiles that will be downloaded.")
        hls_dataset, granules_to_download = create_hls_dataset(
            sub_data_with_tiles, outdir=FLAGS.output_directory
        )
        with open(
            os.path.join(FLAGS.output_directory, "hls_dataset.json"), "w"
        ) as json_file:
            json.dump(hls_dataset, json_file, indent=4)
        pd.DataFrame({"tiles": list(granules_to_download)}).to_csv(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )
    else:
        logging.info("HLS dataset JSON already created")
        with open(
            os.path.join(FLAGS.output_directory, "hls_dataset.json")
        ) as json_file:
            hls_dataset = json.load(json_file)
        granules_to_download = pd.read_csv(
            os.path.join(FLAGS.output_directory, "granules_to_download.csv")
        )["tiles"].tolist()
    os.makedirs(os.path.join(FLAGS.output_directory, "hls_tiles"), exist_ok=True)
    if FLAGS.processing_method in ["download", "download-only"]:
        logging.info("Downloading HLS Tiles")
        parallel_download(
            granules_to_download,
            outdir=os.path.join(FLAGS.output_directory, "hls_tiles"),
        )
    if FLAGS.processing_method == "download-only":
        return
    logging.info("Creating Chips and Segmentation Maps")
    all_chips = []
    all_seg_maps = []
    os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
    os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)
    with dask.distributed.Client() as client:
        client.run(setup)
        for key, hls_tile_dict in tqdm(
            hls_dataset.items(), desc="Processing HLS Dataset"
        ):
            obsv_date_str, tile_id = key.split("_")
            obsv_data = sub_data[
                (sub_data["date"] == pd.to_datetime(obsv_date_str))
                & (sub_data["mgrs_tile_id"].str.contains(tile_id.strip("T")))
            ]
            try:
                chips, seg_maps = create_and_save_chips_with_seg_maps(
                    FLAGS.processing_method,
                    hls_tile_dict,
                    obsv_data,
                    chip_size=FLAGS.chip_size,
                    output_directory=FLAGS.output_directory,
                    no_data_value=FLAGS.no_data_value,
                    src_crs=FLAGS.src_crs,
                    mask_types=FLAGS.mask_types,
                    masking_strategy=FLAGS.masking_strategy,
                    window_size=FLAGS.window_size,
                )
                all_chips.extend(chips)
                all_seg_maps.extend(seg_maps)
            except rasterio.errors.RasterioIOError as e:
                logging.error(
                    f"Error {e} when reading dataset containing: {hls_tile_dict}"
                )
            except IndexError as e:
                logging.error(f"Error {e} when processing {key}")
    logging.info("Saving dataframe of chips and segmentation maps.")
    pd.DataFrame({"Input": all_chips, "Label": all_seg_maps}).to_csv(
        os.path.join(FLAGS.output_directory, "hls_chips_dataset.csv")
    )

    elif FLAGS.data_source == "S2":
        logging.info("Using Sentinel-2 pipeline")

        tile_df, history_dates = get_tile_info(
            sub_data, num_steps=FLAGS.num_steps, temporal_step=FLAGS.temporal_step
        )

        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "s2_dataset.json"))
            and os.path.exists(
                os.path.join(FLAGS.output_directory, "granules_to_download.csv")
            )
        ):
            logging.info("Retrieving Sentinel-2 tiles that will be downloaded.")
            granules_dict = retrieve_sentinel2_metadata(
                tile_df,
                cloud_coverage=FLAGS.cloud_coverage,
                temporal_tolerance=FLAGS.temporal_tolerance,
                history_dates=history_dates,
            )
            logging.info("Creating Sentinel-2 dataset JSON.")
            with open(
                os.path.join(FLAGS.output_directory, "s2_dataset.json"), "w"
            ) as json_file:
                json.dump(granules_dict, json_file, indent=4)
            pd.DataFrame({"tiles": list(granules_dict)}).to_csv(
                os.path.join(FLAGS.output_directory, "granules_to_download.csv")
            )
        else:
            logging.info("Sentinel-2 dataset JSON already created")
            with open(
                os.path.join(FLAGS.output_directory, "s2_dataset.json")
            ) as json_file:
                granules_dict = json.load(json_file)

        logging.info("Downloading Sentinel-2 Tiles")
        download_info_list = download_tile_data(
            granules_dict,
            FLAGS.output_directory,
            client_id=os.getenv("CLIENT_ID"),
            username=os.getenv("USERNAME"),
            password=os.getenv("PASSWORD"),
        )

        if FLAGS.download_only:
            return

        logging.info("Unzipping Sentinel-2 products")
        unzip_all(download_info_list, output_directory=FLAGS.output_directory)

        logging.info("Processing Sentinel-2 products")
        process_tile_bands(granules_dict, output_directory=FLAGS.output_directory)

        for tile_name, tile_data in granules_dict.items():
            filter_best_product_in_folder(
                tile_name,
                tile_data,
                output_directory=FLAGS.output_directory,
                history_dates=history_dates,
                temporal_tolerance=FLAGS.temporal_tolerance,
            )

        logging.info("Creating Chips and Segmentation Maps")

        os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)

        try:
            all_chips, all_seg_maps = create_and_save_chips_with_seg_maps_s2(
                granules_dict=granules_dict,
                sub_data=sub_data,
                chip_size=FLAGS.chip_size,
                output_directory=FLAGS.output_directory,
                no_data_value=FLAGS.no_data_value,
                src_crs=FLAGS.src_crs,
                mask_cloud=FLAGS.mask_cloud,
                water_mask=FLAGS.water_mask,
                temporal_tolerance=FLAGS.temporal_tolerance,
                history_dates=history_dates,
            )

            print(
                f"Generated {len(all_chips)} chips and {len(all_seg_maps)} segmentation maps."
            )
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

        logging.info("Saving dataframe of chips and segmentation maps.")
        pd.DataFrame({"Input": all_chips, "Label": all_seg_maps}).to_csv(
            os.path.join(FLAGS.output_directory, "s2_chips_dataset.csv")
        )

    else:
        raise ValueError(
            "Error: data_source value is not correct. Please enter 'HLS' or 'S2'."
        )


if __name__ == "__main__":
    app.run(main)
