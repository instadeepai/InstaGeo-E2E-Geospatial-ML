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
"""InstaGeo Chip Creator from Raster Module."""

import json
import logging as pylogging
import os
from typing import Any

import geopandas as gpd
import pandas as pd
from absl import app, flags, logging
from pystac_client import Client

from instageo.data import hls_utils
from instageo.data.flags import FLAGS  # Import flags from central location
from instageo.data.hls_utils import HLSRasterPipeline, add_hls_stac_items
from instageo.data.settings import (
    HLSAPISettings,
    HLSBandsSettings,
    HLSBlockSizes,
    NoDataValues,
)

# Create instances of the settings classes
NO_DATA_VALUES = NoDataValues()
HLS_BLOCKSIZE = HLSBlockSizes()
HLS_BANDS = HLSBandsSettings()
HLS_API = HLSAPISettings()

logging.set_verbosity(logging.INFO)
log = pylogging.getLogger(__name__)
log.setLevel(pylogging.WARNING)
pylogging.getLogger("botocore.credentials").setLevel(pylogging.WARNING)
pylogging.getLogger("earthdata").setLevel(pylogging.WARNING)

flags.DEFINE_string(
    "records_file", None, "Path to input records file containing geometries."
)
flags.DEFINE_string("raster_path", None, "Path to input raster file.")


flags.DEFINE_bool(
    "qa_check", True, "Whether to perform quality assurance check on chip and seg_map."
)


def main(argv: Any) -> None:
    """Raster Chip Creator.

    Given raster file containing label information, the Raster Chip Creator creates small chip from
    larger satellite tiles which is suitable for training segmentation models.
    """
    del argv

    if FLAGS.data_source == "HLS":
        if not (
            os.path.exists(os.path.join(FLAGS.output_directory, "hls_dataset.json"))
        ):
            logging.info("Creating HLS dataset JSON.")
            logging.info("Retrieving HLS tile ID for each observation.")
            os.makedirs(os.path.join(FLAGS.output_directory), exist_ok=True)

            obsv_records = gpd.read_file(FLAGS.records_file)
            obsv_records["geometry_4326"] = obsv_records["geometry"].to_crs("EPSG:4326")
            obsv_records["date"] = pd.to_datetime(obsv_records["date"])

            client = Client.open(HLS_API.URL)
            obsv_records_with_hls_items = add_hls_stac_items(
                client,
                obsv_records,
                num_steps=FLAGS.num_steps,
                temporal_step=FLAGS.temporal_step,
                temporal_tolerance=FLAGS.temporal_tolerance,
                cloud_coverage=FLAGS.cloud_coverage,
                daytime_only=FLAGS.daytime_only,
            )
            (
                filtered_obsv_records,
                hls_dataset,
            ) = hls_utils.create_hls_records_with_items(obsv_records_with_hls_items)
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

        hls_raster_pipeline = HLSRasterPipeline(
            output_directory=FLAGS.output_directory,
            chip_size=FLAGS.chip_size,
            raster_path=FLAGS.raster_path,
            mask_types=FLAGS.mask_types,
            masking_strategy=FLAGS.masking_strategy,
            src_crs=FLAGS.src_crs,
            spatial_resolution=FLAGS.spatial_resolution,
            qa_check=FLAGS.qa_check,
            task_type=FLAGS.task_type,
        )

        # Run HLS pipeline
        hls_raster_pipeline.run(hls_dataset, filtered_obsv_records)

    elif FLAGS.data_source == "S2":
        raise NotImplementedError

    elif FLAGS.data_source == "S1":
        raise NotImplementedError

    else:
        raise NotImplementedError


if __name__ == "__main__":
    app.run(main)
