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
from typing import Any

import pandas as pd
import rasterio
from absl import app, flags, logging
from tqdm import tqdm

from instageo.data.geo_utils import get_tiles
from instageo.data.hls_pipeline import (
    add_hls_granules,
    create_and_save_chips_with_seg_maps_hls,
    create_hls_dataset,
    parallel_download,
)
from instageo.data.s2_pipeline import retrieve_sentinel2_metadata

logging.set_verbosity(logging.INFO)

FLAGS = flags.FLAGS

flags.DEFINE_string("dataframe_path", None, "Path to the DataFrame CSV file.")
flags.DEFINE_integer("chip_size", 224, "Size of each chip.")
flags.DEFINE_integer("src_crs", 4326, "CRS of the geo-coordinates in `dataframe_path`")
flags.DEFINE_string(
    "output_directory",
    None,
    "Directory where the chips and segmentation maps will be saved.",
)
flags.DEFINE_integer(
    "no_data_value", -1, "Value to use for no data areas in the segmentation maps."
)
flags.DEFINE_integer("min_count", 100, "Minimum observation counts per tile")
flags.DEFINE_integer("num_steps", 3, "Number of temporal steps")
flags.DEFINE_integer("temporal_step", 30, "Temporal step size.")
flags.DEFINE_integer(
    "temporal_tolerance", 5, "Tolerance used when searching for the closest tile"
)
flags.DEFINE_boolean(
    "download_only", False, "Downloads dataset without creating chips."
)
flags.DEFINE_boolean("mask_cloud", False, "Perform Cloud Masking")
flags.DEFINE_boolean("water_mask", False, "Perform Water Masking")
flags.DEFINE_string(
    "data_source", "HLS", "Data source to use. Accepted values are 'HLS' or 'S2'."
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


def main(argv: Any) -> None:
    """CSV Chip Creator.

    Given a csv file containing geo-located point observations and labels, the Chip
    Creator creates small chip from larger tiles which is suitable for training
    segmentation models.
    """
    del argv
    data = pd.read_csv(FLAGS.dataframe_path)
    data["date"] = pd.to_datetime(data["date"]) - pd.offsets.MonthBegin(1)
    data["input_features_date"] = data["date"] - pd.DateOffset(months=1)
    sub_data = get_tiles(data, min_count=FLAGS.min_count)

    if FLAGS.data_source == "HLS":
        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "hls_dataset.json"))
            and os.path.exists(
                os.path.join(FLAGS.output_directory, "granules_to_download.csv")
            )
        ):
            logging.info("Creating HLS dataset JSON.")
            logging.info("Retrieving HLS tile ID for each observation.")
            sub_data_with_tiles = add_hls_granules(
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
        logging.info("Downloading HLS Tiles")
        parallel_download(
            granules_to_download,
            outdir=os.path.join(FLAGS.output_directory, "hls_tiles"),
        )
        if FLAGS.download_only:
            return
        logging.info("Creating Chips and Segmentation Maps")
        all_chips = []
        all_seg_maps = []
        os.makedirs(os.path.join(FLAGS.output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(FLAGS.output_directory, "seg_maps"), exist_ok=True)
        for key, hls_tile_dict in tqdm(
            hls_dataset.items(), desc="Processing HLS Dataset"
        ):
            obsv_date_str, tile_id = key.split("_")
            obsv_data = sub_data[
                (sub_data["date"] == pd.to_datetime(obsv_date_str))
                & (sub_data["mgrs_tile_id"].str.contains(tile_id.strip("T")))
            ]
            try:
                chips, seg_maps = create_and_save_chips_with_seg_maps_hls(
                    hls_tile_dict,
                    obsv_data,
                    chip_size=FLAGS.chip_size,
                    output_directory=FLAGS.output_directory,
                    no_data_value=FLAGS.no_data_value,
                    src_crs=FLAGS.src_crs,
                    mask_cloud=FLAGS.mask_cloud,
                    water_mask=FLAGS.water_mask,
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
        print("Will use S2 pipeline")

        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "s2_dataset.json"))
            and os.path.exists(
                os.path.join(FLAGS.output_directory, "granules_to_download.csv")
            )
        ):
            logging.info("Creating S2 dataset JSON.")
            logging.info("Retrieving tile ID for each observation.")

            granules_dict = retrieve_sentinel2_metadata(
                sub_data,
                cloud_coverage=FLAGS.cloud_coverage,
                num_steps=FLAGS.num_steps,
                temporal_step=FLAGS.temporal_step,
                temporal_tolerance=FLAGS.temporal_tolerance,
            )

            print(json.dumps(granules_dict, indent=4))

    else:
        raise ValueError(
            "Error: data_source value is not correct. Please enter 'HLS' or 'S2'."
        )


if __name__ == "__main__":
    app.run(main)
