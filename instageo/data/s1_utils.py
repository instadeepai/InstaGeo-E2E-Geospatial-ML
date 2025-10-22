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

"""Utility Functions for generating Sentinel-1 chips."""

import os
import time
from typing import Any

import backoff
import geopandas as gpd
import pandas as pd
import rasterio
import ratelimit
import xarray as xr
from absl import logging
from planetary_computer import sign
from pystac_client import Client

from instageo.data.data_pipeline import (
    NO_DATA_VALUES,
    BasePointsDataPipeline,
    create_segmentation_map,
    get_chip_coords,
    mask_segmentation_map,
)
from instageo.data.settings import (
    DataPipelineSettings,
    S1APISettings,
    S1BandsSettings,
    S1BlockSizes,
)
from instageo.data.stac_utils import (
    find_best_items,
    get_raster_tile_info,
    open_stac_items,
    retrieve_stac_metadata,
)

# Initialize settings
API = S1APISettings()
BANDS = S1BandsSettings()
BLOCKSIZE = S1BlockSizes()
DATA_PIPELINE_SETTINGS = DataPipelineSettings()


def add_s1_stac_items(
    client: Client,
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 12,
    temporal_tolerance_minutes: int = 0,
) -> dict[str, pd.DataFrame]:
    """Searches and adds Sentinel-1 Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the Sentinel-1 granules closest to the
    desired date with a tolerance of `temporal_tolerance`.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest Sentinel-1 items.
        temporal_tolerance_minutes (int): Additional tolerance in minutes for
            finding closest Sentinel-1 items.

    Returns:
        A dictionary mapping each MGRS tile ID to a DataFrame containing the observations
          that fall within that tile with their associated PySTAC items representing granules.
    """
    if "input_features_date" not in data.columns:
        data = data.rename(columns={"date": "input_features_date"})

    tiles_info, tile_queries = get_raster_tile_info(
        data,
        num_steps=num_steps,
        temporal_step=temporal_step,
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
    )

    data["tile_queries"] = tile_queries
    tiles_database = retrieve_stac_metadata(
        client,
        tiles_info,
        collections=API.COLLECTIONS,
        bands_nameplate=BANDS.NAMEPLATE,
        cloud_coverage=None,
        daytime_only=False,
    )
    best_items = find_best_items(
        data,
        tiles_database,
        item_id_field="s1_item_id",
        candidate_items_field="s1_candidate_items",
        items_field="s1_items",
        temporal_tolerance=temporal_tolerance,
    )
    return best_items


class S1PointsPipeline(BasePointsDataPipeline):
    """S1 Points Data Pipeline."""

    def setup(self) -> None:
        """Setup for environment to be used by Dask workers.

        Configures relevant GDAL options for reading COGs
        """
        pass

    @ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.COG_DOWNLOAD_RATELIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (rasterio.errors.RasterioIOError, Exception),
        max_tries=5,
        max_time=300,  # 5 minutes max
        jitter=backoff.full_jitter,
    )
    def load_data(self, tile_dict: dict[str, Any]) -> tuple[xr.Dataset, xr.Dataset, str]:
        """See parent class. Load Granules."""
        try:
            dsb, dsm, crs = open_stac_items(
                tile_dict["granules"],
                self.src_crs,
                self.spatial_resolution,
                bands_asset=BANDS.ASSET,
                blocksize=(BLOCKSIZE.X, BLOCKSIZE.Y),
                mask_band="",
                load_masks=False,
                fill_value=NO_DATA_VALUES.S1,
                sign_func=sign,
            )
        except Exception as e:
            logging.error(f"Error loading data: {str(e)}")
            time.sleep(10 * 60)  # Wait 10 minutes before retrying
            raise

        return dsb, dsm, crs

    def process_tile(
        self,
        obsv_records: gpd.GeoDataFrame,
        flags_dict: dict[str, Any],
        tile_dict: dict[str, Any],
        batch_size: int,
    ) -> tuple[list[str], list[str]]:
        """Processes a single tile.

        Arguments:
            obsv_records: Observation records dataframe.
            flags_dict: Dictionary mapping all arguments and values needed to process the row.
            tile_dict: Input to `self.load_data`, which contains the granules to load.
            batch_size: Number of records to process at a time.
        Returns: tuple of the chip and label filenames lists.
        """
        tile_id = obsv_records.iloc[0]["mgrs_tile_id"]
        stac_items_str = obsv_records.iloc[0]["stac_items_str"]
        chip_paths = []
        label_paths = []

        try:
            date_id = obsv_records.iloc[0]["date"].strftime("%Y%m%d")
            dsb, dsm, crs = self.load_data(tile_dict)
            n_chips_x = dsb.sizes["x"] // self.chip_size
            n_chips_y = dsb.sizes["y"] // self.chip_size
            chip_coords = get_chip_coords(obsv_records, dsb, self.chip_size)

            # Process chips in smaller batches to avoid overwhelming the API
            for i in range(0, len(chip_coords), batch_size):
                batch_coords = chip_coords[i : i + batch_size]
                chips, seg_maps_temp_filenames, chips_temp_filenames = (
                    [],
                    [],
                    [],
                )

                for x, y in batch_coords:
                    # TODO: handle potential partially out of bound chips
                    if (x >= n_chips_x) or (y >= n_chips_y):
                        continue

                    chip_id = f"{date_id}_{tile_id}_{x}_{y}"
                    chip_name = f"chip_{chip_id}.tif"
                    seg_map_name = f"seg_map_{chip_id}.tif"

                    chip_filename = os.path.join(self.output_directory, "chips", chip_name)
                    chips_temp_filenames.append(chip_filename)
                    seg_map_filename = os.path.join(self.output_directory, "seg_maps", seg_map_name)
                    seg_maps_temp_filenames.append(seg_map_filename)
                    if os.path.exists(chip_filename) or os.path.exists(seg_map_filename):
                        logging.info(f"Skipping {chip_filename} because it's already created")
                        continue

                    chip = dsb.isel(
                        x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                        y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                    )
                    chips.append(chip)

                # Process the batch
                try:
                    # Compute chips locally before processing
                    chips = [chip.compute() for chip in chips]

                    for chip, chip_filename, seg_map_filename in zip(
                        chips, chips_temp_filenames, seg_maps_temp_filenames
                    ):
                        if chip.where(chip != NO_DATA_VALUES.S1).count().values == 0:
                            logging.warning(f"Skipping {chip_filename} due to no valid data pixels")
                            continue

                        seg_map = create_segmentation_map(
                            chip, obsv_records, self.window_size, self.task_type
                        )
                        seg_map = mask_segmentation_map(
                            chip,
                            seg_map,
                            NO_DATA_VALUES.S1,
                            self.masking_strategy,
                        )

                        if seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP).count().values == 0:
                            logging.warning(f"Skipping {seg_map_filename} due to empty label")
                            continue

                        label_paths.append(seg_map_filename)
                        chip_paths.append(chip_filename)
                        seg_map.rio.to_raster(seg_map_filename)
                        chip.rio.to_raster(chip_filename)

                except Exception as e:
                    logging.error(f"Error processing batch: {str(e)}")
                    continue

                # Add a delay between batches to avoid rate limiting
                time.sleep(5)

        except rasterio.errors.RasterioIOError as e:
            logging.error(f"Error {e} when reading dataset containing: {stac_items_str}")
        except Exception as e:
            logging.error(f"Error {e} when processing {stac_items_str}")

        return chip_paths, label_paths
