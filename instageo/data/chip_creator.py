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
import logging as std_logging
import os
from functools import partial
from typing import Any

import dask.distributed
import earthaccess
import pandas as pd
import rasterio
from absl import app, flags, logging
from dotenv import load_dotenv
from tqdm import tqdm

from instageo.data import hls_utils, s1_utils, s2_utils
from instageo.data.data_pipeline import (
    MASK_DECODING_POS,
    NO_DATA_VALUES,
    apply_mask,
    create_and_save_chips_with_seg_maps,
    get_pystac_client,
    get_tiles,
)
from instageo.data.settings import GDALOptions

load_dotenv(os.path.expanduser("~/.credentials"))
logging.set_verbosity(logging.INFO)
std_logging.getLogger("botocore.credentials").setLevel(logging.WARNING)

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
    will be made to retrieve `num_steps` chips prior to the observation date.
    Otherwise, the value of `num_steps` will default to 1 and an attempt will be made to retrieve
    the chip corresponding to the observation date.
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
flags.DEFINE_enum("data_source", "HLS", ["HLS", "S2", "S1"], "Data source to use.")
flags.DEFINE_integer(
    "cloud_coverage",
    10,
    "Percentage of cloud cover to use. Accepted values are between 0.0001 and 100.",
    lower_bound=0,
    upper_bound=100,
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
    - "download" corresponds to downloading entire tiles to be used for creating
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
    lambda val_list: all(v in MASK_DECODING_POS["HLS"].keys() for v in val_list),
    message=f"Valid values are {list(MASK_DECODING_POS['HLS'].keys())}",
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
    """CSV Chip Creator.

    Given a csv file containing geo-located point observations and labels, the Chip
    Creator creates small chip from larger tiles which is suitable for training
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
    sub_data = get_tiles(data, src_crs=FLAGS.src_crs, min_count=FLAGS.min_count)

    if FLAGS.data_source == "HLS":
        logging.info("Using Harmonized Landsat Sentinel-2 pipeline")
        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "hls_dataset.json"))
            and os.path.exists(
                os.path.join(FLAGS.output_directory, "hls_granules_to_download.csv")
            )
        ):
            logging.info("Creating HLS dataset JSON.")
            logging.info("Retrieving HLS tile ID for each observation.")
            sub_data_with_tiles = hls_utils.add_hls_granules(
                sub_data,
                num_steps=FLAGS.num_steps,
                temporal_step=FLAGS.temporal_step,
                temporal_tolerance=FLAGS.temporal_tolerance,
                cloud_coverage=FLAGS.cloud_coverage,
            )
            logging.info("Retrieving HLS tiles that will be downloaded.")
            hls_dataset, hls_granules_to_download = hls_utils.create_hls_dataset(
                sub_data_with_tiles, outdir=FLAGS.output_directory
            )
            with open(
                os.path.join(FLAGS.output_directory, "hls_dataset.json"), "w"
            ) as json_file:
                json.dump(hls_dataset, json_file, indent=4)
            pd.DataFrame({"tiles": list(hls_granules_to_download)}).to_csv(
                os.path.join(FLAGS.output_directory, "hls_granules_to_download.csv")
            )
        else:
            logging.info("HLS dataset JSON already created")
            with open(
                os.path.join(FLAGS.output_directory, "hls_dataset.json")
            ) as json_file:
                hls_dataset = json.load(json_file)
            hls_granules_to_download = pd.read_csv(
                os.path.join(FLAGS.output_directory, "hls_granules_to_download.csv")
            )["tiles"].tolist()
        os.makedirs(os.path.join(FLAGS.output_directory, "hls_tiles"), exist_ok=True)
        logging.info("Downloading HLS Tiles")
        if FLAGS.processing_method in ["download", "download-only"]:
            logging.info("Downloading HLS Tiles")
            hls_utils.parallel_download(
                hls_granules_to_download,
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
                        data_reader=(
                            hls_utils.open_hls_cogs
                            if FLAGS.processing_method == "cog"
                            else hls_utils.open_mf_tiff_dataset
                        ),
                        mask_fn=apply_mask,
                        processing_method=FLAGS.processing_method,
                        tile_dict=hls_tile_dict,
                        data_source=FLAGS.data_source,
                        df=obsv_data,
                        chip_size=FLAGS.chip_size,
                        output_directory=FLAGS.output_directory,
                        no_data_value=NO_DATA_VALUES.get("HLS"),
                        src_crs=FLAGS.src_crs,
                        mask_decoder=hls_utils.decode_fmask_value,
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
                            partial(s2_utils.search_and_open_s2_cogs, s2_pystac_client)
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
                        no_data_value=NO_DATA_VALUES.get("S2"),
                        src_crs=FLAGS.src_crs,
                        mask_decoder=s2_utils.create_mask_from_scl,
                        mask_types=FLAGS.mask_types,
                        masking_strategy=FLAGS.masking_strategy,
                        window_size=FLAGS.window_size,
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
                        no_data_value=NO_DATA_VALUES.get("S1"),
                        src_crs=FLAGS.src_crs,
                        mask_decoder=s1_utils.decode_mask,
                        mask_types=[],
                        masking_strategy=FLAGS.masking_strategy,
                        window_size=FLAGS.window_size,
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
