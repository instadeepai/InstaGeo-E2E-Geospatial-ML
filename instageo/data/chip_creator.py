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

import ast
import json
import logging as std_logging
import os
from functools import partial
from typing import Any, List, Tuple

import dask.distributed
import earthaccess
import geopandas as gpd
import pandas as pd
import rasterio
from absl import app, flags, logging
from dotenv import load_dotenv
from pystac_client import Client
from shapely.geometry import Point
from tqdm import tqdm

from instageo.data import hls_utils, s1_utils, s2_utils
from instageo.data.data_pipeline import (
    NO_DATA_VALUES,
    apply_mask,
    create_and_save_chips_with_seg_maps,
    get_pystac_client,
    get_tiles,
)
from instageo.data.flags import FLAGS
from instageo.data.hls_utils import HLSPointsPipeline, add_hls_stac_items
from instageo.data.settings import GDALOptions, S2Bands

load_dotenv(os.path.expanduser("~/.credentials"))
logging.set_verbosity(logging.INFO)
std_logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

S2_BANDS = S2Bands().VALUES

flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame file.")

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
flags.DEFINE_list(
    "s2_bands",
    ["B02", "B03", "B04", "B8A", "B11", "B12"],
    "List of different bands to extract. Applies only if `data_source` is set to 'S2'",
)
flags.register_validator(
    "s2_bands",
    lambda bands: all(b in S2_BANDS for b in bands),
    message=f"Valid values are {S2_BANDS}",
)

flags.DEFINE_enum(
    "processing_method",
    "cog",
    ["cog", "download", "download-only"],
    """Method to use to process the tiles:
    - "cog" corresponds to creating the chips by utilizing the Cloud Optimized GeoTIFFs.
    - "download" corresponds to downloading entire tiles to be used for creating
    the chips.
    - "download-only" corresponds to a simple download of the tiles without further
    processing.""",
)
flags.DEFINE_enum(
    "data_format",
    "csv",
    ["csv", "parquet"],
    """Format of the original file containing the observations. The data must contain
    columns named 'date', 'x', 'y', 'label'.
    In case of a Parquet file, the partitions must be done by the 'year' and 'mgrs_tile_id'
    columns.
    """,
)
flags.DEFINE_string(
    "filters",
    None,
    """List of filters to use. Filters must be provided as tuples following the
    structure ('col_to_filter_on' ? 'operator' ? value). Applies only in case of
    Parquet files.
    - The columns on which to filter and the operators should be provided as strings.
    - The operators allowed are ['==', '=', '>', '>=', '<', '<=', '!=', 'in', 'not in']
    Example: "('year' ? '==' ? 2016); ('mgrs_tile_id' ? '!=' ? 'BAN')"
    "('year' ? 'in' ? [2016, 2020]); ('mgrs_tile_id' ? 'not in' ? ['13SCS', 'BAN'])"
    """,
)


def parse_tuple_list(flag_value: str) -> List[Tuple]:
    """Converts a string into a list of tuples.

    Args:
        flag_value (str): String containing values to parse as list of tuples.
            Each tuple is separated by ';'. Within each tuple values are separated
            by '?'.

    Returns:
        (List[Tuple]): List of tuples extracted from the original string.
    """
    try:
        return [tuple(item.strip("()").split("?")) for item in flag_value.split(";")]
    except Exception as e:
        raise ValueError(
            f"Error parsing string {flag_value} to extract filters list: {e}"
        )


def parse_filters(flag_value: str) -> List[Tuple[str, str, Any]]:
    """Converts a list of tuples into valid filters.

    Args:
        flag_value (str): String containing values to parse as list of tuples.
            Each tuple is separated by ';'. Within each tuple values are separated
            by '?'.

    Returns:
        (List[Tuple[str, str, Any]]): A parsed version of the list of tuples to be used as filters.
    """
    try:
        filters = parse_tuple_list(flag_value)
        parsed_filters = []
        ops = ["==", "=", ">", ">=", "<", "<=", "!=", "in", "not in"]
        for filter in filters:
            col, op, val = filter
            try:
                col = ast.literal_eval(col)
                op = ast.literal_eval(op)
                val = ast.literal_eval(val)
            except Exception as e:
                raise flags.ValidationError(
                    f"Could not properly parse filter {filter}: {e}"
                )
            if not isinstance(col, str):
                raise flags.ValidationError("Provide the filter column as a string")
            if op not in ops:
                raise flags.ValidationError(f"Operators must be one of {ops}")
            parsed_filters.append((col, op, val))
    except Exception as e:
        raise flags.ValidationError(
            f"Filters must be provided as tuple ('col_to_filter_on' ? 'operator' ? value): {e}"
        )
    return parsed_filters


# TODO: Use flags.mark_flags_as_required
def check_required_flags() -> None:
    """Check if required flags are provided."""
    required_flags = ["dataframe_path", "output_directory"]
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise app.UsageError(f"Flag --{flag_name} is required.")


def setup() -> None:
    """Setup for environment to be used by Dask client.

    Configures relevant GDAL options for reading COGs
    """
    earthaccess.login(strategy="environment", persist=True)
    env = rasterio.Env(**GDALOptions().model_dump())
    env.__enter__()


def main(argv: Any) -> None:
    """CSV/Parquet Chip Creator.

    Given a file containing geo-located point observations and labels, the Chip
    Creator creates small chip from larger tiles which is suitable for training
    segmentation models.
    """
    del argv
    if FLAGS.data_format == "parquet":
        try:
            filters = None
            if FLAGS.filters:
                filters = parse_filters(FLAGS.filters)
            data = pd.read_parquet(
                FLAGS.dataframe_path,
                engine="pyarrow",
                filters=filters,
            )
        except Exception as e:
            raise ValueError(f"Provide valid path and filters: {e}")
    else:
        data = pd.read_csv(FLAGS.dataframe_path)

    # Convert date column to datetime
    data["date"] = pd.to_datetime(data["date"])

    # If time column exists, combine it with date
    # Time format expected is HH:MM:SS
    if "time" in data.columns:
        data["date"] = data["date"] + pd.to_timedelta(data["time"])

    # Shift to month start if requested
    if FLAGS.shift_to_month_start:
        data["date"] = data["date"] - pd.offsets.MonthBegin(1)

    data["input_features_date"] = (
        data["date"] - pd.DateOffset(days=FLAGS.temporal_step)
        if FLAGS.is_time_series_task
        else data["date"]
    )
    FLAGS.num_steps = 1 if not FLAGS.is_time_series_task else FLAGS.num_steps
    sub_data = get_tiles(data, src_crs=FLAGS.src_crs, min_count=FLAGS.min_count)

    if FLAGS.data_source == "HLS":
        logging.info("Using Harmonized Landsat Sentinel-2 pipeline")
        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "hls_dataset.json"))
            and os.path.exists(
                os.path.join(FLAGS.output_directory, "filtered_obsv_records.gpkg")
            )
        ):
            logging.info("Creating HLS dataset JSON.")
            logging.info("Retrieving HLS tile ID for each observation.")
            os.makedirs(os.path.join(FLAGS.output_directory), exist_ok=True)

            # Convert to GeoPandas DataFrame
            # Once all pipelines have been migrated, move this section outside this if-block
            geometry = [Point(xy) for xy in zip(sub_data["x"], sub_data["y"])]
            sub_data = gpd.GeoDataFrame(
                sub_data, geometry=geometry, crs=f"EPSG:{FLAGS.src_crs}"
            )
            sub_data["geometry_4326"] = sub_data["geometry"].to_crs("EPSG:4326")

            client = Client.open(hls_utils.API.URL)
            sub_data_with_hls_items = add_hls_stac_items(
                client,
                sub_data,
                num_steps=FLAGS.num_steps,
                temporal_step=FLAGS.temporal_step,
                temporal_tolerance=FLAGS.temporal_tolerance,
                temporal_tolerance_minutes=FLAGS.temporal_tolerance_minutes,
                cloud_coverage=FLAGS.cloud_coverage,
                daytime_only=FLAGS.daytime_only,
            )
            (
                filtered_obsv_records,
                hls_dataset,
            ) = hls_utils.create_hls_records_with_items(sub_data_with_hls_items)
            with open(
                os.path.join(FLAGS.output_directory, "hls_dataset.json"), "w"
            ) as json_file:
                json.dump(hls_dataset, json_file, indent=4)
            filtered_obsv_records.to_file(
                os.path.join(FLAGS.output_directory, "filtered_obsv_records.gpkg"),
                driver="GPKG",
            )
        else:
            logging.info("HLS dataset JSON already created")
            with open(
                os.path.join(FLAGS.output_directory, "hls_dataset.json")
            ) as json_file:
                hls_dataset = json.load(json_file)
            filtered_obsv_records = gpd.read_file(
                os.path.join(FLAGS.output_directory, "filtered_obsv_records.gpkg")
            )

        if FLAGS.processing_method in ["download", "download-only"]:
            logging.info("Downloading HLS Tiles")
            os.makedirs(
                os.path.join(FLAGS.output_directory, "hls_tiles"), exist_ok=True
            )
            hls_utils.parallel_download(
                hls_dataset,
                outdir=os.path.join(FLAGS.output_directory, "hls_tiles"),
            )
        if FLAGS.processing_method == "download-only":
            return
        logging.info("Creating Chips and Segmentation Maps")

        hls_points_pipeline = HLSPointsPipeline(
            output_directory=FLAGS.output_directory,
            chip_size=FLAGS.chip_size,
            mask_types=FLAGS.mask_types,
            masking_strategy=FLAGS.masking_strategy,
            src_crs=FLAGS.src_crs,
            spatial_resolution=FLAGS.spatial_resolution,
            window_size=FLAGS.window_size,
        )

        # Run HLS Points pipeline
        hls_points_pipeline.run(hls_dataset, filtered_obsv_records)

    elif FLAGS.data_source == "S2":
        logging.info("Using Sentinel-2 pipeline")
        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "s2_dataset.json"))
            and os.path.exists(
                os.path.join(FLAGS.output_directory, "s2_granules_to_download.csv")
            )
        ):
            logging.info("Creating S2 dataset JSON.")
            logging.info("Retrieving S2 tile ID for each observation.")

            sub_data_with_tiles = s2_utils.add_s2_granules(
                sub_data,
                num_steps=FLAGS.num_steps,
                temporal_step=FLAGS.temporal_step,
                temporal_tolerance=FLAGS.temporal_tolerance,
                cloud_coverage=FLAGS.cloud_coverage,
            )

            logging.info("Retrieving S2 tiles that will be downloaded.")
            s2_dataset, s2_granules_to_download = s2_utils.create_s2_dataset(
                sub_data_with_tiles, outdir=FLAGS.output_directory
            )
            with open(
                os.path.join(FLAGS.output_directory, "s2_dataset.json"), "w"
            ) as json_file:
                json.dump(s2_dataset, json_file, indent=4)
            s2_granules_to_download.to_csv(
                os.path.join(FLAGS.output_directory, "s2_granules_to_download.csv")
            )
        else:
            logging.info("S2 dataset JSON already created")
            with open(
                os.path.join(FLAGS.output_directory, "s2_dataset.json")
            ) as json_file:
                s2_dataset = json.load(json_file)
            s2_granules_to_download = pd.read_csv(
                os.path.join(FLAGS.output_directory, "s2_granules_to_download.csv")
            )
        s2_tiles_dir = os.path.join(FLAGS.output_directory, "s2_tiles")
        os.makedirs(s2_tiles_dir, exist_ok=True)
        if FLAGS.processing_method in ["download", "download-only"]:
            logging.info("Downloading Sentinel-2 Tiles")
            s2_utils.download_tile_data(
                s2_granules_to_download,
                s2_tiles_dir,
                client_id=os.getenv("CLIENT_ID"),
                username=os.getenv("USERNAME"),
                password=os.getenv("PASSWORD"),
            )
            logging.info("Unzipping Sentinel-2 products")
            s2_utils.extract_and_delete_zip_files(s2_tiles_dir)

        if FLAGS.processing_method == "download-only":
            return

        logging.info("Creating Chips and Segmentation Maps")
        all_chips = []
        all_seg_maps = []
        os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)
        s2_pystac_client = get_pystac_client()
        with dask.distributed.Client() as client:
            for key, tile_dict in tqdm(
                s2_dataset.items(), desc="Processing Sentinel-2 Dataset"
            ):
                obsv_date_str, tile_id = key.split("_")
                obsv_data = sub_data[
                    (sub_data["date"] == pd.to_datetime(obsv_date_str))
                    & (sub_data["mgrs_tile_id"].str.contains(tile_id.strip("T")))
                ]
                try:
                    chips, seg_maps = create_and_save_chips_with_seg_maps(
                        data_reader=(
                            partial(
                                s2_utils.search_and_open_s2_cogs,
                                s2_pystac_client,
                                FLAGS.s2_bands,
                            )
                            if FLAGS.processing_method == "cog"
                            else s2_utils.open_mf_jp2_dataset
                        ),
                        mask_fn=apply_mask,
                        processing_method=FLAGS.processing_method,
                        tile_dict=tile_dict,
                        data_source=FLAGS.data_source,
                        df=obsv_data,
                        chip_size=FLAGS.chip_size,
                        output_directory=FLAGS.output_directory,
                        no_data_value=NO_DATA_VALUES.S2,
                        src_crs=FLAGS.src_crs,
                        mask_decoder=s2_utils.create_mask_from_scl,
                        mask_types=FLAGS.mask_types,
                        masking_strategy=FLAGS.masking_strategy,
                        window_size=FLAGS.window_size,
                        task_type=FLAGS.task_type,
                    )
                    all_chips.extend(chips)
                    all_seg_maps.extend(seg_maps)
                except rasterio.errors.RasterioIOError as e:
                    logging.error(
                        f"Error {e} when reading dataset containing: {tile_dict}"
                    )
                except IndexError as e:
                    logging.error(f"Error {e} when processing {key}")

        logging.info("Saving dataframe of chips and segmentation maps.")
        pd.DataFrame({"Input": all_chips, "Label": all_seg_maps}).to_csv(
            os.path.join(FLAGS.output_directory, "s2_chips_dataset.csv")
        )

    elif FLAGS.data_source == "S1":
        logging.info("Using Sentinel-1 pipeline")
        s1_pystac_client = get_pystac_client()
        s1_dataset_with_items: Any = {}
        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "s1_dataset.json"))
        ):
            logging.info("Creating S1 dataset JSON.")
            logging.info("Retrieving S1 tile ID for each observation.")

            sub_data_with_tiles = s1_utils.add_s1_items(
                s1_pystac_client,
                sub_data,
                src_crs=FLAGS.src_crs,
                num_steps=FLAGS.num_steps,
                temporal_step=FLAGS.temporal_step,
                temporal_tolerance=FLAGS.temporal_tolerance,
            )

            logging.info("Retrieving S1 tiles that will be loaded.")
            s1_dataset, s1_dataset_with_items = s1_utils.create_s1_dataset(
                sub_data_with_tiles
            )
            with open(
                os.path.join(FLAGS.output_directory, "s1_dataset.json"), "w"
            ) as json_file:
                json.dump(s1_dataset, json_file, indent=4)
        else:
            logging.info("S1 dataset JSON already created")
            with open(
                os.path.join(FLAGS.output_directory, "s1_dataset.json")
            ) as json_file:
                s1_dataset = json.load(json_file)
                logging.info("Loading PySTAC items from dataset")
                s1_dataset_with_items = s1_utils.load_pystac_items_from_dataset(
                    s1_dataset
                )
        logging.info("Creating Chips and Segmentation Maps")
        all_chips = []
        all_seg_maps = []
        os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)
        FLAGS.processing_method = "cog"
        with dask.distributed.Client() as client:
            for key, tile_dict in tqdm(
                s1_dataset_with_items.items(), desc="Processing Sentinel-1 Dataset"
            ):
                obsv_date_str, tile_id, _ = key.split("_")
                obsv_data = sub_data[
                    (sub_data["date"] == pd.to_datetime(obsv_date_str))
                    & (sub_data["mgrs_tile_id"].str.contains(tile_id.strip("T")))
                ]
                try:
                    chips, seg_maps = create_and_save_chips_with_seg_maps(
                        data_reader=s1_utils.open_s1_cogs,
                        mask_fn=apply_mask,
                        processing_method=FLAGS.processing_method,
                        tile_dict=tile_dict,
                        data_source=FLAGS.data_source,
                        df=obsv_data,
                        chip_size=FLAGS.chip_size,
                        output_directory=FLAGS.output_directory,
                        no_data_value=NO_DATA_VALUES.S1,
                        src_crs=FLAGS.src_crs,
                        mask_decoder=s1_utils.decode_mask,
                        mask_types=[],
                        masking_strategy=FLAGS.masking_strategy,
                        window_size=FLAGS.window_size,
                        task_type=FLAGS.task_type,
                    )
                    all_chips.extend(chips)
                    all_seg_maps.extend(seg_maps)
                except rasterio.errors.RasterioIOError as e:
                    logging.error(
                        f"Error {e} when reading dataset containing: {tile_dict}"
                    )
                except IndexError as e:
                    logging.error(f"Error {e} when processing {key}")

        logging.info("Saving dataframe of chips and segmentation maps.")
        pd.DataFrame({"Input": all_chips, "Label": all_seg_maps}).to_csv(
            os.path.join(FLAGS.output_directory, "s1_chips_dataset.csv")
        )
    else:
        raise ValueError(
            "Error: data_source value is not correct. Please enter 'HLS' or 'S2' or 'S1'."
        )


if __name__ == "__main__":
    app.run(main)
