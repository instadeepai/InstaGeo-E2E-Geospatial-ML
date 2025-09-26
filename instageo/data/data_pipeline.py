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

"""InstaGeo Data pipeline Module."""

import logging
import os
import time
from abc import ABC, abstractmethod
from functools import partial
from typing import Any, Callable, Dict, List, Tuple

import dask
import dask.distributed
import geopandas as gpd
import mgrs
import numpy as np
import pandas as pd
import rasterio  # noqa: F401
import xarray as xr
from pyproj import Transformer
from pystac_client import Client
from tqdm import tqdm

from instageo.data.settings import DataPipelineSettings, NoDataValues

# Masks decoding positions
MASK_DECODING_POS: dict[str, dict] = {
    "HLS": {"cloud": 1, "near_cloud_or_shadow": 2, "cloud_shadow": 3, "water": 5},
    "S2": {"cloud": [8, 9], "water": [6]},
}

# No data values
NO_DATA_VALUES = NoDataValues()
DATA_PIPELINE_SETTINGS = DataPipelineSettings()

# Microsoft Planetary Computer STAC API
MPC_STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def mask_segmentation_map(
    chip: xr.DataArray,
    seg_map: xr.DataArray,
    chip_no_data_value: xr.DataArray,
    masking_strategy: str = "any",
) -> xr.DataArray:
    """Masks segmentation map.

    Checks for chip_no_data_value in the chip and masks the segmentation values
    that correspond to no data value in the chip (at least for one band).

    Args:
        seg_map (DataArray): Segmentation map to mask
        chip (DataArray): Chip that correspond to the segmentation map
        chip_no_data_value (int): Value to use for no data areas in the chips.
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)

    Returns:
        The segmentation map after masking
    """
    if masking_strategy == "each":
        valid_mask = (chip != chip_no_data_value).any(dim="band").astype(np.uint16)
    elif masking_strategy == "any":
        valid_mask = (chip != chip_no_data_value).all(dim="band").astype(np.uint16)
    else:
        raise ValueError(f"Invalid masking strategy: {masking_strategy}")
    seg_no_data_value = NO_DATA_VALUES.SEG_MAP
    seg_map = seg_map.where(valid_mask, seg_no_data_value)
    return seg_map


def create_and_save_chips_with_seg_maps(
    data_reader: Callable | partial,
    mask_fn: Callable,
    processing_method: str,
    tile_dict: dict[str, Any],
    data_source: str,
    df: pd.DataFrame,
    chip_size: int,
    output_directory: str,
    no_data_value: int,
    src_crs: int,
    mask_decoder: Callable,
    mask_types: list[str],
    masking_strategy: str,
    window_size: int,
    task_type: str = "seg",
) -> tuple[list[str], list[str | None]]:
    """Chip Creator.

    Create chips and corresponding segmentation maps from a satellite image tile and save
    them to an output directory.

    Args:
        data_reader (callable[dict[str, Any], bool] | functools.partial): A multi-file reader that
            accepts a dictionary of satellite image tile paths and reads it into an Xarray dataset
            or dataarray. Optionally performs masking based on the boolean mask types provided.
        mask_fn (Callable): Function to use to apply masks.
        processing_method (str): Processing method to use to create the chips and
        segmentation maps.
        tile_dict (Dict): A dict mapping band names to tile filepath.
        data_source (str): Data source, which can be "HLS", "S2" or "S1".
        df (pd.DataFrame): DataFrame containing the data for segmentation maps.
        chip_size (int): Size of each chip.
        output_directory (str): Directory where the chips and segmentation maps will be
            saved.
        no_data_value (int): Value to use for no data areas in the chips.
        src_crs (int): CRS of points in `df`
        mask_types (list[str]): Types of masking to perform.
        mask_decoder (Callable): Function to use to process/extract actual mask values
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)
        window_size (int): Window size to use around the observation pixel.
        task_type (str): Task type to use to adjust the data type of the segmentation maps.

    Returns:
        A tuple containing the lists of created chips and segmentation maps.
    """
    load_masks = True if mask_types else False
    dsb, dsm, crs = data_reader(tile_dict, load_masks=load_masks)
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
    # TODO: handle chip names more gracefully
    if data_source == "HLS":
        tile_name_splits = tile_dict["tiles"]["B02_0"].split(".")
        tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
    elif data_source == "S2":
        tile_name_splits = tile_dict["granules"][0].split(".")[0].split("/")[-1].split("_")
        tile_id = (
            f"{tile_name_splits[0]}_{tile_name_splits[1]}_"
            f"{tile_name_splits[5]}_{tile_name_splits[2]}"
        )
    elif data_source == "S1":
        tile_name_splits = tile_dict["items"][0].id.split("_")
        tile_id = "_".join(tile_name_splits[0:2] + [tile_name_splits[4]] + tile_name_splits[6:9])

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
            chip = mask_fn(
                chip=chip,
                mask=chip_mask,
                no_data_value=no_data_value,
                mask_decoder=mask_decoder,
                data_source=data_source,
                mask_types=mask_types,
                masking_strategy=masking_strategy,
            )
        if chip.where(chip != no_data_value).count().values == 0:
            continue
        seg_map = create_segmentation_map(chip, df, window_size, task_type)
        seg_map = mask_segmentation_map(chip, seg_map, no_data_value)
        seg_no_data_value = NO_DATA_VALUES.SEG_MAP
        if seg_map.where(seg_map != seg_no_data_value).count().values == 0:
            continue

        seg_maps.append(seg_map_name)
        seg_map.rio.to_raster(seg_map_filename)
        chips.append(chip_name)
        chip.rio.to_raster(chip_filename)
    return chips, seg_maps


def apply_mask(
    chip: xr.DataArray,
    mask: xr.DataArray,
    no_data_value: int,
    mask_decoder: Callable,
    data_source: str,
    masking_strategy: str = "each",
    mask_types: list[str] = list(MASK_DECODING_POS["HLS"].keys()),
) -> xr.DataArray:
    """Apply masking to a chip.

    Args:
        chip (xr.DataArray): Chip array containing the pixels to be masked out.
        mask (xr.DataArray): Array containing the masks.
        no_data_value (int): Value to be used for masked pixels.
        mask_decoder (Callable): Function to use to process/extract actual mask values
        data_source (str): Data source used to extract masking positions based on mask types
        masking_strategy (str): Masking strategy to apply ("each" for timestep-wise masking,
        and "any" to exclude pixels if the mask is present for at least one timestep. The
        behavior is the same if the chip is extracted for one timestep.)
        mask_types (list[str]): Mask types to apply.

    Returns:
        xr.DataArray: The masked data array.
    """
    for mask_type in mask_types:
        pos = MASK_DECODING_POS[data_source].get(mask_type, None)
        if pos:
            decoded_mask = mask_decoder(mask, pos)
            if masking_strategy == "each":
                # repeat across timesteps so that, each mask is applied to its
                # corresponding timestep
                decoded_mask = decoded_mask.values.repeat(chip.shape[0] // mask.shape[0], axis=0)
            elif masking_strategy == "any":
                # collapse the mask to exclude a pixel if its corresponding mask value
                # for at least one timestep is 1
                decoded_mask = decoded_mask.values.any(axis=0)
            chip = chip.where(decoded_mask == 0, other=no_data_value)
    return chip


def get_tile_info(
    data: pd.DataFrame,
    num_steps: int = 3,
    temporal_step: int = 10,
    temporal_tolerance: int = 5,
    temporal_tolerance_minutes: int = 0,
) -> tuple[pd.DataFrame, list[tuple[str, list[str]]]]:
    """Get Tile Info.

    Retrieves a summary of all tiles required for a given dataset. The summary contains
    the desired start and end date for each tile. Also retrieves a list of queries
    that can be used to retrieve the tiles for each observation in `data`.

    Args:
        data (pd.DataFrame): A dataframe containing observation records.
        num_steps (int): Number of temporal time steps
        temporal_step (int): Size of each temporal step.
        temporal_tolerance (int): Number of days used as offset for the
        start and end dates to search for each tile.
        temporal_tolerance_minutes (int): Number of minutes to add to the temporal
            tolerance.

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    push_max_date_to_end_of_day = "time" not in data.columns
    data = data[["mgrs_tile_id", "input_features_date", "x", "y"]].reset_index(drop=True)
    tile_queries = []
    tile_info: Any = []
    for _, (tile_id, date, lon, lat) in data.iterrows():
        history = []
        for i in range(num_steps):
            curr_date = date - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%dT%H:%M:%S"))
            tile_info.append([tile_id, curr_date, lon, lat])
        tile_queries.append((tile_id, history))
    tile_info = (
        pd.DataFrame(tile_info, columns=["tile_id", "date", "lon", "lat"])
        .groupby("tile_id")
        .agg(
            min_date=("date", "min"),
            max_date=("date", "max"),
            lon_min=("lon", "min"),
            lon_max=("lon", "max"),
            lat_min=("lat", "min"),
            lat_max=("lat", "max"),
        )
    ).reset_index()

    total_temporal_tol = temporal_tolerance + (temporal_tolerance_minutes / (24 * 60))
    tile_info["min_date"] -= pd.Timedelta(days=total_temporal_tol)
    tile_info["max_date"] += pd.Timedelta(days=total_temporal_tol)
    tile_info["min_date"] = tile_info["min_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    if push_max_date_to_end_of_day:
        tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%dT23:59:59")
    else:
        tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%dT%H:%M:%S")
    return tile_info, tile_queries


def reproject_coordinates(df: pd.DataFrame, source_epsg: int = 4326) -> pd.DataFrame:
    """Reproject coordinates from the source EPSG to EPSG:4326.

    This function reprojects the geo coordinates found in df dataframe to the EPSG:4326

    Args:
        df (pd.DataFrame): DataFrame containing longitude and latitude columns.
        source_epsg (int): The EPSG code of the source CRS for invalid coordinates.

    Returns:
        pd.DataFrame: DataFrame with transformed and valid coordinates.
    """
    logging.info("Reprojecting coordinates to EPSG:4326...")
    transformer = Transformer.from_crs(f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True)

    # Reproject the invalid rows using Vectorized transformation
    x, y = transformer.transform(df["x"].values, df["y"].values)
    df[["x", "y"]] = np.column_stack((x, y))

    return df


def get_tiles(data: pd.DataFrame, src_crs: int = 4326, min_count: int = 100) -> pd.DataFrame:
    """Retrieve Tile IDs for Geospatial Observations from Satellite Data.

    This function associates each geospatial observation with a tile ID based on its
    geographic location, accommodating datasets with varying density across locations. By
    focusing on more densely populated areas, it enables more efficient resource usage and
    refined data analysis.

    The function assigns a tile ID to each observation, counts the occurrences within
    each tile, and retains only those tiles with a specified minimum count (`min_count`) of
    observations.

    Args:
        data: DataFrame containing geospatial observations with location coordinates.
        src_crs (int): CRS of points in `data`
        min_count: Minimum count of observations required per tile to retain.

    Returns:
        A subset of observations within tiles that meet or exceed the specified `min_count`.
    """
    if src_crs != 4326:
        data = reproject_coordinates(data, source_epsg=src_crs)
    if "mgrs_tile_id" not in data.columns:
        mgrs_object = mgrs.MGRS()
        get_mgrs_tile_id = lambda row: mgrs_object.toMGRS(row["y"], row["x"], MGRSPrecision=0)
        data["mgrs_tile_id"] = data.apply(get_mgrs_tile_id, axis=1)
    tile_counts = data.groupby("mgrs_tile_id").size().sort_values(ascending=False)
    data = pd.merge(data, tile_counts.reset_index(name="counts"), how="left", on="mgrs_tile_id")
    sub_data = data[data["counts"] >= min_count]
    assert not sub_data.empty, "No observation records left"
    return sub_data


def create_segmentation_map(
    chip: Any, df: pd.DataFrame, window_size: int, task_type: str = "seg"
) -> xr.DataArray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        window_size (int): Window size to use around the observation pixel.
        task_type (str): Task type to use to adjust the data type of the segmentation maps.

    Returns:
         xr.DataArray: The created segmentation map as an xarray DataArray.
    """
    seg_map = xr.full_like(
        chip.isel(band=0),
        fill_value=NO_DATA_VALUES.SEG_MAP,
        dtype=np.int16 if task_type == "seg" else np.float32,
    )
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
    window_rows = np.clip(rows[:, np.newaxis, np.newaxis] + offset_rows, 0, chip.sizes["x"] - 1)
    window_cols = np.clip(cols[:, np.newaxis, np.newaxis] + offset_cols, 0, chip.sizes["y"] - 1)
    window_labels = np.repeat(df.label.values, offset_rows.ravel().shape)
    seg_map.values[window_rows.ravel(), window_cols.ravel()] = window_labels
    return seg_map


def get_chip_coords(df: gpd.GeoDataFrame, tile: xr.DataArray, chip_size: int) -> np.array:
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


def get_pystac_client() -> Client:
    """Opens a pystac_client Client instance using MPC STAC API URL.

    Returns:
        Client : A client with an established connection to the STAC Catalog.
    """
    return Client.open(MPC_STAC_API_URL)


def adjust_dims(data: xr.DataArray) -> xr.DataArray:
    """Adjusts dimensions of a dataarray.

    This function stacks the "time" and "band" dims over a new "band" dim and reorders
    the dataarray dims into ("band","y","x").

    Args:
        data (xr.DataArray): A dataarray for which dimensions need to be adjusted.

    Returns:
        xr.DataArray: A 3D xarray DataArray without 'time' dimension.
    """
    num_bands = data["band"].size
    data = data.stack(time_band=("time", "band"))
    new_bands_indices = [
        f"{band}_{i // num_bands}" for i, (_, band) in enumerate(data.coords["time_band"].values)
    ]
    data = data.drop_vars(["time_band", "time", "band"])
    data.coords["time_band"] = new_bands_indices
    data = data.rename({"time_band": "band"}).transpose("band", "y", "x")
    return data


class BaseRasterDataPipeline(ABC):
    """Abstract Base Class for Raster-Based Data Pipelines.

    This class defines the structure for geospatial data processing pipelines that
    work with raster imagery and associated masks. It is designed to support large-scale,
    distributed processing using Dask, and to streamline the generation of data chips
    and segmentation labels for machine learning applications.
    """

    def __init__(
        self,
        output_directory: str,
        chip_size: int,
        raster_path: str,
        mask_types: List[str],
        masking_strategy: str,
        src_crs: int,
        spatial_resolution: float,
        qa_check: bool = True,
        task_type: str = "seg",
        is_bbox_feature: bool = False,
    ) -> None:
        """Init."""
        self.output_directory = output_directory
        self.chip_size = chip_size
        self.raster_path = raster_path
        self.mask_types = mask_types
        self.masking_strategy = masking_strategy
        self.src_crs = src_crs
        self.spatial_resolution = spatial_resolution
        self.qa_check = qa_check
        self.task_type = task_type
        self.is_bbox_feature = is_bbox_feature

    @abstractmethod
    def setup(self) -> None:
        """Set up necessary configurations on Dask workers.

        Configures relevant GDAL options for reading COGs.
        """
        pass

    @abstractmethod
    def load_data(self, tile_dict: Dict[str, Any]) -> Tuple[xr.Dataset, xr.Dataset, str]:
        """Loads data for a specific tile.

        Arguments:
            A dictionary with a `granules` key. The value of the key should be a
            list of STAC items.

        Returns:
            After loading the granules, it should return a tuple containing the following:
                - Xarray dataset containing data from the granules
                - Xarray dataset containing the mask information
                - CRS string
        """
        pass

    @abstractmethod
    def process_row(
        self,
        row_dict: Dict[str, Any],
        flags_dict: Dict[str, Any],
        tile_dict: dict[str, Any],
    ) -> Tuple[str, str | None] | None:
        """Processes a single row of data.

        Arguments:
            row_dict: Dictionary created from a row in the observation records dataframe.
            flags_dict: Dictionary mapping all arguments and values needed to process the row.
            tile_dict: Dictionary containing granules STAC items.

        Returns: None if successful or a tuple of the chip and label filenames.
        """
        pass

    def _process_batch(
        self,
        client: dask.distributed.Client,
        dataset: Dict[str, Any],
        batch_records: pd.DataFrame,
        flags_dict: Dict[str, Any],
    ) -> List[Tuple[str, str]]:
        """Process a batch of records in parallel.

        Args:
            client: Dask client for parallel processing
            dataset: Dataset dictionary
            batch_records: Batch of records to process
            flags_dict: Dictionary of flags for processing

        Returns:
            List of (chip_path, label_path) tuples
        """
        futures = []
        for _, row in batch_records.iterrows():
            row_dict = row.to_dict()
            row_dict["geometry"] = row.geometry.__geo_interface__
            label_filename = (
                f"{os.path.splitext(row_dict['label_filename'])[0]}_" f"{row_dict['mgrs_tile_id']}"
            )
            chip_filename = label_filename.replace("mask", "merged").replace("label", "chip")

            chip_path = os.path.join(self.output_directory, "chips", f"{chip_filename}.tif")
            if os.path.exists(chip_path):
                logging.info(f"Skipping {chip_path} because it's already created")
                continue

            futures.append(
                client.submit(
                    self.process_row,
                    row_dict,
                    flags_dict,
                    dataset[row_dict["stac_items_str"]],
                )
            )

        results = client.gather(futures)
        return [result for result in results if result is not None]

    def run(self, dataset: Dict[str, Any], obsv_records: pd.DataFrame) -> None:
        """Main method to run the pipeline and create all chips and corresponding labels.

        Arguments:
            dataset: A dataset mapping `key` to STAC Items.
            obsv_records: A dataframe containing a column that match each row to STAC items in
                `dataset` using `key`

        Returns:
            None.
        """
        with dask.distributed.Client() as client:
            with dask.distributed.performance_report(
                filename=os.path.join(self.output_directory, "dask-report.html")
            ):
                client.run(self.setup)
                logging.info(f"View Dask Distributed Dashboard at {client.dashboard_link}.")
                os.makedirs(os.path.join(self.output_directory, "chips"), exist_ok=True)
                os.makedirs(os.path.join(self.output_directory, "seg_maps"), exist_ok=True)

                # Prepare flags for serialization
                flags_dict = {
                    "output_directory": self.output_directory,
                    "chip_size": self.chip_size,
                    "mask_types": self.mask_types,
                    "masking_strategy": self.masking_strategy,
                }

                chip_paths = []
                label_paths = []
                batch_size = DATA_PIPELINE_SETTINGS.BATCH_SIZE
                total_records = len(obsv_records)

                for i in tqdm(
                    range(0, total_records, batch_size),
                    desc="Processing batches",
                    total=total_records // batch_size,
                ):
                    batch_records = obsv_records.iloc[i : i + batch_size]
                    try:
                        results = self._process_batch(client, dataset, batch_records, flags_dict)
                        for chip_path, label_path in results:
                            chip_paths.append(chip_path)
                            label_paths.append(label_path)
                        time.sleep(5)  # Add delay between batches
                    except Exception as e:
                        logging.error(f"Error processing batch {i // batch_size}: {str(e)}")
                        time.sleep(2 * 60)  # Wait 2 minutes before retrying
                        continue

        logging.info("Saving dataframe of chips and segmentation maps.")
        if self.is_bbox_feature:
            chips_df = pd.DataFrame({"Input": chip_paths})
        else:
            chips_df = pd.DataFrame({"Input": chip_paths, "Label": label_paths})
        chips_df.to_csv(os.path.join(self.output_directory, "hls_raster_dataset.csv"))


class BasePointsDataPipeline(ABC):
    """Data Pipeline Abstract Base Class."""

    def __init__(
        self,
        output_directory: str,
        chip_size: int,
        mask_types: List[str],
        masking_strategy: str,
        src_crs: int,
        spatial_resolution: float,
        qa_check: bool = True,
        window_size: int = 0,
    ) -> None:
        """Init."""
        self.output_directory = output_directory
        self.chip_size = chip_size
        self.mask_types = mask_types
        self.masking_strategy = masking_strategy
        self.src_crs = src_crs
        self.spatial_resolution = spatial_resolution
        self.qa_check = qa_check
        self.window_size = window_size

    @abstractmethod
    def setup(self) -> None:
        """Set up necessary configurations on Dask workers.

        Configures relevant GDAL options for reading COGs.
        """
        pass

    def _is_stac_item_processed(
        self, stac_items_str: str, obsv_records: pd.DataFrame, existing_chips: set[str]
    ) -> bool:
        """Check if any chips and segmentation maps for a STAC item have been processed.

        Args:
            stac_items_str: The STAC items string identifier
            obsv_records: DataFrame containing observation records
            existing_chips: Set of already processed chip identifiers

        Returns:
            bool: True if any chips and segmentation maps exist, False otherwise
        """
        df = obsv_records[obsv_records["stac_items_str"] == stac_items_str]
        if df.empty:
            return False

        # Get the first record to construct the chip identifier
        first_record = df.iloc[0]
        date_id = first_record["date"].strftime("%Y%m%d")
        tile_name_splits = stac_items_str.split("_")[0].split(".")
        tile_id = f"{tile_name_splits[1]}_{tile_name_splits[2]}_{tile_name_splits[3]}"
        chip_base_id = f"{date_id}_{tile_id}"

        return chip_base_id in existing_chips

    @abstractmethod
    def load_data(self, tile_dict: Dict[str, Any]) -> Tuple[xr.Dataset, xr.Dataset, str]:
        """Loads data for a specific tile.

        Arguments:
            A dictionary with a `granules` key. The value of the key should be a
            list of STAC items.

        Returns:
            After loading the granules, it should return a tuple containing the following:
                - Xarray dataset containing data from the granules
                - Xarray dataset containing the mask information
                - CRS string
        """
        pass

    @abstractmethod
    def process_tile(
        self,
        obsv_records: gpd.GeoDataFrame,
        flags_dict: Dict[str, Any],
        tile_dict: Dict[str, Any],
        batch_size: int,
    ) -> Tuple[list[str], list[str]]:
        """Processes a single HLS granule.

        Arguments:
            obsv_records: Observation records dataframe.
            flags_dict: Dictionary mapping all arguments and values needed to process the row.
            tile_dict: Dictionary containing granules STAC items.
            batch_size: Number of records to process at a time.
        Returns: A list of the chip and label filenames.
        """
        pass

    def run(self, dataset: Dict[str, Any], obsv_records: pd.DataFrame) -> None:
        """Main method to run the pipeline and create all chips and corresponding labels.

        Arguments:
            dataset: A dataset mapping `key` to STAC Items.
            obsv_records: A dataframe containing a column that match each row to STAC items in
                `dataset` using `key`

        Returns:
            None.
        """
        # Create output directories
        os.makedirs(os.path.join(self.output_directory, "chips"), exist_ok=True)
        os.makedirs(os.path.join(self.output_directory, "seg_maps"), exist_ok=True)

        # Collect existing chip identifiers
        existing_chips = set()
        for filename in os.listdir(os.path.join(self.output_directory, "chips")):
            if filename.startswith("chip_") and filename.endswith(".tif"):
                # Extract the base identifier (date_tile_id) from the filename
                base_id = "_".join(
                    filename.split("_")[1:-2]
                )  # Remove 'chip_' prefix and x_y.tif suffix
                existing_chips.add(base_id)

        # Filter out already processed STAC items
        dataset = {
            stac_items_str: tile_dict
            for stac_items_str, tile_dict in dataset.items()
            if not self._is_stac_item_processed(stac_items_str, obsv_records, existing_chips)
        }

        if not dataset:
            logging.info("All STAC items have already been processed. Nothing to do.")
            return

        with dask.distributed.Client() as client:
            with dask.distributed.performance_report(
                filename=os.path.join(self.output_directory, "dask-report.html")
            ):
                client.run(self.setup)
                logging.info(f"View Dask Distributed Dashboard at {client.dashboard_link}.")

                # Prepare flags for serialization
                flags_dict = {
                    "output_directory": self.output_directory,
                    "chip_size": self.chip_size,
                    "mask_types": self.mask_types,
                    "masking_strategy": self.masking_strategy,
                }
                chip_paths = []
                label_paths = []
                for stac_items_str, tile_dict in tqdm(
                    dataset.items(), desc="Processing HLS Dataset"
                ):
                    df = obsv_records[obsv_records["stac_items_str"] == stac_items_str]
                    future = client.submit(
                        self.process_tile,
                        df,
                        flags_dict,
                        tile_dict,
                        batch_size=DATA_PIPELINE_SETTINGS.BATCH_SIZE,
                    )
                    result = future.result()
                    if result is not None:
                        chip_paths.extend(result[0])
                        label_paths.extend(result[1])
        logging.info("Saving dataframe of chips and segmentation maps.")
        pd.DataFrame({"Input": chip_paths, "Label": label_paths}).to_csv(
            os.path.join(self.output_directory, "hls_raster_dataset.csv")
        )
