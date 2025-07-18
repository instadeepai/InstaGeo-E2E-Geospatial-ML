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

"""Utility Functions for Reading and Processing Harmonized Landsat Sentinel-2 Dataset."""

import logging
import os
import time
from typing import Any, List

import backoff
import earthaccess
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import ratelimit
import rioxarray  # noqa
import stackstac
import xarray as xr
from pystac.item import Item
from pystac.item_collection import ItemCollection
from pystac_client import Client
from rasterio.crs import CRS
from shapely.geometry import shape

from instageo.data import geo_utils, hls_utils
from instageo.data.data_pipeline import (
    BasePointsDataPipeline,
    BaseRasterDataPipeline,
    adjust_dims,
    apply_mask,
    create_segmentation_map,
    get_chip_coords,
    mask_segmentation_map,
)
from instageo.data.settings import (
    DataPipelineSettings,
    GDALOptions,
    HLSAPISettings,
    HLSBandsSettings,
    HLSBlockSizes,
    NoDataValues,
)
from instageo.data.stac_utils import (
    find_best_items,
    get_raster_tile_info,
    retrieve_stac_metadata,
)

# Create instances of the settings classes
NO_DATA_VALUES = NoDataValues()
BLOCKSIZE = HLSBlockSizes()
BANDS = HLSBandsSettings()
API = HLSAPISettings()
DATA_PIPELINE_SETTINGS = DataPipelineSettings()

client = Client.open(API.URL)


def decode_fmask_value(
    value: xr.Dataset | xr.DataArray, position: int
) -> xr.Dataset | xr.DataArray:
    """Decodes HLS v2.0 Fmask.

    Returns:
        Xarray dataset containing decoded bits.
    """
    quotient = value // (2**position)
    return quotient - ((quotient // 2) * 2)


def open_mf_tiff_dataset(
    band_files: dict[str, Any], load_masks: bool
) -> tuple[xr.Dataset, xr.Dataset | None, CRS]:
    """Open multiple TIFF files as an xarray Dataset.

    Args:
        band_files (Dict[str, Dict[str, str]]): A dictionary mapping band names to file paths.
        load_masks (bool): Whether or not to load the masks files.

    Returns:
        (xr.Dataset, xr.Dataset | None, CRS): A tuple of xarray Dataset combining data from all the
            provided TIFF files, (optionally) the masks, and the CRS
    """
    band_paths = list(band_files["tiles"].values())
    bands_dataset = xr.open_mfdataset(
        band_paths,
        concat_dim="band",
        combine="nested",
        mask_and_scale=False,  # Scaling will be applied manually
    )
    bands_dataset.band_data.attrs["scale_factor"] = 1
    mask_paths = list(band_files["fmasks"].values())
    mask_dataset = (
        xr.open_mfdataset(
            mask_paths,
            concat_dim="band",
            combine="nested",
        )
        if load_masks
        else None
    )
    with rasterio.open(band_paths[0]) as src:
        crs = src.crs
    return bands_dataset, mask_dataset, crs


def parallel_download(
    dataset: dict[str, Any], outdir: str, max_retries: int = 3
) -> None:
    """Parallel Download.

    Wraps `download_tile` with multiprocessing.Pool for downloading multiple tiles in
    parallel.

    Args:
        dataset: A dataset mapping `key` to STAC Items.
        outdir: Directory to save downloaded tiles.
        max_retries: Number of times to retry downloading all tiles.

    Returns:
        None
    """
    num_cpus = os.cpu_count()
    earthaccess.login(persist=True)
    retries = 0
    complete = False

    urls = set()
    for _, dataset_entry in dataset.items():
        stac_items = dataset_entry["granules"]
        for item in stac_items:
            urls.update([item["assets"][band]["href"] for band in BANDS.ASSET])

    while retries <= max_retries:
        temp_urls = [
            url
            for url in urls
            if not os.path.exists(os.path.join(outdir, url.split("/")[-1]))
        ]
        if not temp_urls:
            complete = True
            break
        earthaccess.download(temp_urls, local_path=outdir, threads=num_cpus)
        for filename in os.listdir(outdir):
            file_path = os.path.join(outdir, filename)
            if os.path.isfile(file_path):
                file_size = os.path.getsize(file_path)
                if file_size < 1024:
                    os.remove(file_path)
        retries += 1
    if complete:
        logging.info("Successfully downloaded all granules")
    else:
        logging.warning(
            f"Couldn't download the following granules after {max_retries} retries:\n{urls}"  # noqa
        )


def dispatch_hls_candidate_items(
    tile_observations: gpd.GeoDataFrame,
    tile_candidate_items: List[Item],
) -> pd.DataFrame | None:
    """Dispatches appropriate HLS PySTAC items to each observation.

    A given observation will have a candidate item if it's geometry falls within the
    geometry of the granule.

    Args:
        tile_observations (pandas.DataFrame): DataFrame containing observations
            of the same tile.
        tile_candidate_items (List[Item]): List of candidate items for a given tile
    Returns:
        A DataFrame with observations and their containing granules (items)
        when possible.
    """
    hls_item_ids = [item.id for item in tile_candidate_items]
    candidate_items_gdf = gpd.GeoDataFrame.from_features(tile_candidate_items, crs=4326)
    candidate_items_gdf["hls_item_id"] = hls_item_ids
    candidate_items_gdf = candidate_items_gdf[["hls_item_id", "geometry"]]

    tile_observations = tile_observations.set_geometry("geometry_4326")

    matches = gpd.sjoin(
        tile_observations,
        candidate_items_gdf,
        predicate="within",
    )
    if matches.empty:
        return None
    else:
        tile_observations["hls_candidate_items"] = (
            matches.groupby(matches.index)
            .agg(
                {
                    "index_right": (
                        lambda indices: [tile_candidate_items[id] for id in indices]
                    )
                }
            )
            .reindex(tile_observations.index, fill_value=[])
        )
        return tile_observations


def add_hls_stac_items(
    client: Client,
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 12,
    temporal_tolerance_minutes: int = 0,
    cloud_coverage: int = 10,
    daytime_only: bool = False,
) -> dict[str, pd.DataFrame]:
    """Searches and adds HLS Granules.

    Data contains tile_id and a series of date for which the tile is desired. This
    function takes the tile_id and the dates and finds the HLS granules with the least cloud
    coverage closest to the desired date with a tolerance of `temporal_tolerance`.

    Args:
        client (pystac_client.Client): pystac_client client to use to perform the search.
        data (pd.DataFrame): A dataframe containing observations that fall within a
            dense tile.
        num_steps (int): Number of temporal steps into the past to fetch.
        temporal_step (int): Step size (in days) for creating temporal steps.
        temporal_tolerance (int): Tolerance (in days) for finding closest HLS items.
        temporal_tolerance_minutes (int): Additional tolerance in minutes for finding
            closest HLS items.
        cloud_coverage (int): Maximum percentage of cloud coverage to be tolerated for a granule.
        daytime_only (bool): Flag to determine whether to filter out night time granules.

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
        cloud_coverage=cloud_coverage,
        daytime_only=daytime_only,
    )
    best_items = find_best_items(
        data,
        tiles_database,
        item_id_field="hls_item_id",
        candidate_items_field="hls_candidate_items",
        items_field="hls_items",
        temporal_tolerance=temporal_tolerance,
        temporal_tolerance_minutes=temporal_tolerance_minutes,
    )
    return best_items


def is_valid_dataset_entry(obsv: pd.Series) -> bool:
    """Checks HLS granules validity for a given observation.

    The granules will be added to the dataset if they are all non
    null and unique for all timesteps.

    Args:
        obsv (pandas.Series): Observation data for which to assess the validity

    Returns:
        True if the granules are unique and non null.
    """
    if any(granule is None for granule in obsv["hls_granules"]) or (
        len(obsv["hls_granules"]) != len(set(obsv["hls_granules"]))
    ):
        return False
    return True


class HLSRasterPipeline(BaseRasterDataPipeline):
    """HLS Raster Data Pipeline."""

    def setup(self) -> None:
        """Setup for environment to be used by Dask workers.

        Configures relevant GDAL options for reading COGs
        """
        earthaccess.login(persist=True)
        env = rasterio.Env(**GDALOptions().model_dump())
        env.__enter__()

    @ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.COG_DOWNLOAD_RATELIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (rasterio.errors.RasterioIOError, Exception),
        max_tries=5,
        max_time=300,  # 5 minutes max
        jitter=backoff.full_jitter,
    )
    def load_data(
        self, tile_dict: dict[str, Any]
    ) -> tuple[xr.DataArray, xr.DataArray, str]:
        """See parent class. Load Granules."""
        dsb, dsm, crs = open_hls_stac_items(
            tile_dict["granules"],
            self.src_crs,
            self.spatial_resolution,
            load_masks=True,
            fill_value=NO_DATA_VALUES.HLS,
        )
        return dsb, dsm, crs

    def process_row(
        self,
        row_dict: dict[str, Any],
        flags_dict: dict[str, Any],
        tile_dict: dict[str, Any],
    ) -> None | tuple[str, str | None]:
        """See parent class. Process a single row."""
        label_filename = f"{os.path.splitext(row_dict['label_filename'])[0]}_{row_dict['mgrs_tile_id']}"  # noqa
        chip_filename = label_filename.replace("mask", "merged").replace(
            "label", "chip"
        )

        chip_path = os.path.join(self.output_directory, "chips", f"{chip_filename}.tif")

        label_path = os.path.join(
            self.output_directory, "seg_maps", f"{label_filename}.tif"
        )
        if os.path.exists(chip_path) and os.path.exists(label_path):
            logging.info(f"Skipping {chip_path} because it's already created")
            return chip_path, label_path
        if os.path.exists(chip_path) and self.is_bbox_feature:
            logging.info(f"Skipping {chip_path} because it's already created")
            return chip_path, None
        try:
            dsb, dsm, _ = self.load_data(tile_dict)
            geometry = shape(row_dict["geometry"])

            # Process chip
            chip = geo_utils.slice_xr_dataset(dsb, geometry, chip_size=self.chip_size)
            if not self.is_bbox_feature:
                seg_map = xr.open_dataarray(
                    os.path.join(self.raster_path, row_dict["label_filename"])
                )
            else:
                seg_map = None

            if dsm is not None:
                chip_mask = geo_utils.slice_xr_dataset(
                    dsm, geometry, chip_size=self.chip_size
                )
                chip = apply_mask(
                    chip=chip,
                    mask=chip_mask,
                    no_data_value=0,
                    mask_decoder=hls_utils.decode_fmask_value,
                    data_source="HLS",
                    mask_types=self.mask_types,
                    masking_strategy=self.masking_strategy,
                )

            if chip is not None and seg_map is None:
                # Clip values to valid HLS range (0-10000)
                chip = chip.clip(min=0, max=10000)
                chip = chip.where(~np.isnan(chip), NO_DATA_VALUES.HLS).astype(np.uint16)
                chip.squeeze().rio.to_raster(chip_path)
                return chip_path, None
            elif (
                chip is not None
                and chip.sizes["x"] == seg_map.sizes["x"]
                and chip.sizes["y"] == seg_map.sizes["y"]
            ):
                # Overrides the chip coordinates to match the segmentation map.
                seg_map, chip = xr.align(
                    seg_map, chip, join="override", exclude=["band"]
                )

                # Clip values to valid HLS range (0-10000)
                chip = chip.clip(min=0, max=10000)

                if self.qa_check:
                    if chip.where(chip != NO_DATA_VALUES.HLS).count().values == 0:
                        logging.warning(f"Skipping {chip_filename} due to cloud")
                        return None
                    seg_map = mask_segmentation_map(
                        chip, seg_map, NO_DATA_VALUES.HLS, self.masking_strategy
                    )
                    if (
                        seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP).count().values
                        == 0
                    ):
                        logging.warning(f"Skipping {label_filename} due to empty label")
                        return None
                seg_map = seg_map.where(
                    ~np.isnan(seg_map), NO_DATA_VALUES.SEG_MAP
                ).astype(np.uint8 if self.task_type == "seg" else np.float32)
                chip = chip.where(~np.isnan(chip), NO_DATA_VALUES.HLS).astype(np.uint16)

                seg_map.squeeze().rio.to_raster(label_path)
                chip.squeeze().rio.to_raster(chip_path)
                return chip_path, label_path
            else:
                logging.warning(f"Skipping {label_filename} due to invalid shapes")
            return None
        except Exception as e:
            logging.error(f"Error processing row {row_dict}: {str(e)}")
            return None


class HLSPointsPipeline(BasePointsDataPipeline):
    """HLS Raster Data Pipeline."""

    def setup(self) -> None:
        """Setup for environment to be used by Dask workers.

        Configures relevant GDAL options for reading COGs
        """
        earthaccess.login(persist=True)
        env = rasterio.Env(**GDALOptions().model_dump())
        env.__enter__()

    @ratelimit.limits(calls=DATA_PIPELINE_SETTINGS.COG_DOWNLOAD_RATELIMIT, period=60)
    @backoff.on_exception(
        backoff.expo,
        (rasterio.errors.RasterioIOError, Exception),
        max_tries=5,
        max_time=300,  # 5 minutes max
        jitter=backoff.full_jitter,
    )
    def load_data(
        self, tile_dict: dict[str, Any]
    ) -> tuple[xr.Dataset, xr.Dataset, str]:
        """See parent class. Load Granules."""
        dsb, dsm, crs = open_hls_stac_items(
            tile_dict["granules"],
            self.src_crs,
            self.spatial_resolution,
            load_masks=True,
        )
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
        tile_name_splits = tile_dict["granules"][0]["id"].split(".")
        tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
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
                chips, masks, seg_maps_temp_filenames, chips_temp_filenames = (
                    [],
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

                    chip_filename = os.path.join(
                        self.output_directory, "chips", chip_name
                    )
                    chips_temp_filenames.append(chip_filename)
                    seg_map_filename = os.path.join(
                        self.output_directory, "seg_maps", seg_map_name
                    )
                    seg_maps_temp_filenames.append(seg_map_filename)
                    if os.path.exists(chip_filename) or os.path.exists(
                        seg_map_filename
                    ):
                        logging.info(
                            f"Skipping {chip_filename} because it's already created"
                        )
                        continue

                    chip = dsb.isel(
                        x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                        y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                    )
                    chips.append(chip)

                    if dsm is not None:
                        chip_mask = dsm.isel(
                            x=slice(x * self.chip_size, (x + 1) * self.chip_size),
                            y=slice(y * self.chip_size, (y + 1) * self.chip_size),
                        )
                        masks.append(chip_mask)
                    else:
                        masks.append(None)

                # Process the batch
                try:
                    # Compute chips and masks locally before processing
                    chips = [chip.compute() for chip in chips]
                    masks = (
                        [mask.compute() for mask in masks] if dsm is not None else masks
                    )

                    for chip, mask, chip_filename, seg_map_filename in zip(
                        chips, masks, chips_temp_filenames, seg_maps_temp_filenames
                    ):
                        if mask is not None:
                            chip = apply_mask(
                                chip=chip,
                                mask=mask,
                                no_data_value=NO_DATA_VALUES.HLS,
                                mask_decoder=decode_fmask_value,
                                data_source="HLS",
                                mask_types=self.mask_types,
                                masking_strategy=self.masking_strategy,
                            )

                        if chip.where(chip != NO_DATA_VALUES.HLS).count().values == 0:
                            logging.warning(f"Skipping {chip_filename} due to cloud")
                            continue

                        seg_map = create_segmentation_map(
                            chip, obsv_records, self.window_size
                        )
                        seg_map = mask_segmentation_map(
                            chip,
                            seg_map,
                            NO_DATA_VALUES.HLS,
                            self.masking_strategy,
                        )

                        if (
                            seg_map.where(seg_map != NO_DATA_VALUES.SEG_MAP)
                            .count()
                            .values
                            == 0
                        ):
                            logging.warning(
                                f"Skipping {seg_map_filename} due to empty label"
                            )
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
            logging.error(
                f"Error {e} when reading dataset containing: {stac_items_str}"
            )
        except Exception as e:
            logging.error(f"Error {e} when processing {stac_items_str}")

        return chip_paths, label_paths


def open_hls_stac_items(
    tile_dict: dict[str, Any],
    epsg: int,
    resolution: float,
    load_masks: bool = False,
    fill_value: int = 0,
) -> tuple[xr.DataArray, xr.DataArray | None, str]:
    """Opens multiple HLS STAC Items as an xarray DataArray from given granules `tile_dict`.

    Args:
        tile_dict (dict[str, Any]): A dictionary containing granules IDs to retrieve
        for all timesteps of interest.
        epsg (int): CRS EPSG code.
        resolution (float): Spatial resolution in the specified CRS.
        load_masks (bool): Whether or not to load the masks COGs.
        fill_value (int): Fill value for the data array.

    Returns:
        (xr.DataArray, xr.DataArray | None, str): A tuple of xarray DataArray combining
        data from all the COGs bands of interest, (optionally) the COGs masks and the
        CRS used.
    """
    # Load the bands for all timesteps and stack them in a data array
    assets_to_load = BANDS.ASSET + ["Fmask"] if load_masks else BANDS.ASSET

    # Convert items to plain dicts before stacking to avoid STAC catalog resolution
    plain_items = [item.to_dict() for item in ItemCollection(tile_dict)]

    stacked_items = stackstac.stack(
        plain_items,
        assets=assets_to_load,
        chunksize=(BLOCKSIZE.X, BLOCKSIZE.Y),
        properties=False,
        rescale=False,
        fill_value=fill_value,
        epsg=epsg,
        resolution=resolution,
    )

    bands = adjust_dims(stacked_items.sel(band=BANDS.ASSET))
    masks = adjust_dims(stacked_items.sel(band=["Fmask"])) if load_masks else None

    bands = bands.astype(np.uint16)
    bands.attrs["scale_factor"] = 1

    return bands, masks, bands.crs
