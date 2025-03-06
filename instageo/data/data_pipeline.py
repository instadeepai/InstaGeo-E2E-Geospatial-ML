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
from functools import partial
from typing import Any, Callable

import geopandas as gpd
import mgrs
import numpy as np
import pandas as pd
import xarray as xr
from pyproj import Transformer
from pystac_client import Client
from shapely.geometry import box

from instageo.data.settings import NoDataValues

# Masks decoding positions
MASK_DECODING_POS: dict[str, dict] = {
    "HLS": {"cloud": 1, "water": 5},
    "S2": {"cloud": [8, 9], "water": [6]},
}

# No data values
NO_DATA_VALUES = NoDataValues().model_dump()

# Microsoft Planetary Computer STAC API
MPC_STAC_API_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def mask_segmentation_map(
    chip: xr.DataArray, seg_map: xr.DataArray, no_data_value: xr.DataArray
) -> xr.DataArray:
    """Masks segmentation map.

    Checks for no_data_value in the chip and masks the segmentation values
    that correspond to no data value in the chip (at least for one band).

    Args:
        seg_map (DataArray): Segmentation map to mask
        chip (DataArray): Chip that correspond to the segmentation map
        no_data_value (int): Value to use for no data areas in the chips.

    Returns:
        The segmentation map after masking
    """
    valid_mask = (chip != no_data_value).all(dim="band").astype(np.uint8)
    seg_no_data_value = NO_DATA_VALUES.get("SEG_MAP")
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
        tile_name_splits = (
            tile_dict["granules"][0].split(".")[0].split("/")[-1].split("_")
        )
        tile_id = (
            f"{tile_name_splits[0]}_{tile_name_splits[1]}_"
            f"{tile_name_splits[5]}_{tile_name_splits[2]}"
        )
    elif data_source == "S1":
        tile_name_splits = tile_dict["items"][0].id.split("_")
        tile_id = "_".join(
            tile_name_splits[0:2] + [tile_name_splits[4]] + tile_name_splits[6:9]
        )

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
        seg_map = create_segmentation_map(chip, df, window_size)
        seg_map = mask_segmentation_map(chip, seg_map, no_data_value)
        seg_no_data_value = NO_DATA_VALUES.get("SEG_MAP")
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
                decoded_mask = decoded_mask.values.repeat(
                    chip.shape[0] // mask.shape[0], axis=0
                )
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

    Returns:
        A `tile_info` dataframe and a list of `tile_queries`
    """
    data = data[["mgrs_tile_id", "input_features_date", "x", "y"]].reset_index(
        drop=True
    )
    tile_queries = []
    tile_info: Any = []
    for _, (tile_id, date, lon, lat) in data.iterrows():
        history = []
        for i in range(num_steps):
            curr_date = date - pd.Timedelta(days=temporal_step * i)
            history.append(curr_date.strftime("%Y-%m-%d"))
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
    tile_info["min_date"] -= pd.Timedelta(days=temporal_tolerance)
    tile_info["max_date"] += pd.Timedelta(days=temporal_tolerance)
    tile_info["min_date"] = tile_info["min_date"].dt.strftime("%Y-%m-%d")
    tile_info["max_date"] = tile_info["max_date"].dt.strftime("%Y-%m-%d")
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
    transformer = Transformer.from_crs(
        f"EPSG:{source_epsg}", "EPSG:4326", always_xy=True
    )

    # Reproject the invalid rows
    df[["x", "y"]] = df.apply(
        lambda row: transformer.transform(row["x"], row["y"]), axis=1
    )

    return df


def get_tiles(
    data: pd.DataFrame, src_crs: int = 4326, min_count: int = 100
) -> pd.DataFrame:
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
    mgrs_object = mgrs.MGRS()
    get_mgrs_tile_id = lambda row: mgrs_object.toMGRS(
        row["y"], row["x"], MGRSPrecision=0
    )
    data["mgrs_tile_id"] = data.apply(get_mgrs_tile_id, axis=1)
    tile_counts = data.groupby("mgrs_tile_id").size().sort_values(ascending=False)
    data = pd.merge(
        data, tile_counts.reset_index(name="counts"), how="left", on="mgrs_tile_id"
    )
    sub_data = data[data["counts"] >= min_count]
    assert not sub_data.empty, "No observation records left"
    return sub_data


def create_segmentation_map(
    chip: Any, df: pd.DataFrame, window_size: int
) -> xr.DataArray:
    """Create a segmentation map for the chip using the DataFrame.

    Args:
        chip (Any): The chip (subset of the original data) for which the segmentation
            map is being created.
        df (pd.DataFrame): DataFrame containing the data to be used in the segmentation
            map.
        window_size (int): Window size to use around the observation pixel.

    Returns:
         xr.DataArray: The created segmentation map as an xarray DataArray.
    """
    seg_map = xr.full_like(
        chip.isel(band=0), fill_value=NO_DATA_VALUES.get("SEG_MAP"), dtype=np.int16
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


def make_valid_bbox(
    lon_min: float, lat_min: float, lon_max: float, lat_max: float
) -> tuple[float, float, float, float]:
    """Create a valid bounding box to search for tiles.

    The purpose of this function is to still be able to extract data through
    earthaccess even given just a single observation in a tile (min_count = 1).
    When the number of observations in a tile is 1, or if we only have aligned
    observations, the lon_min, lat_min, lon_max, lat_max extracted from those
    won't produce a valid bounding box. Thus, we attempt to create a small buffer
    around the observation(s) to produce a valid bounding box.

    Args:
        lon_min (float): Minimum longitude
        lat_min (float): Minimum latitude
        lon_max (float): Maximum longitude
        lat_max (float): Maximum latitude

    Returns:
        A tuple of coordinates to use for a bounding box

    """
    epsilon = 1e-3
    if box(lon_min, lat_min, lon_max, lat_max).is_valid:
        return lon_min, lat_min, lon_max, lat_max
    else:
        return box(lon_min, lat_min, lon_max, lat_max).buffer(epsilon).bounds


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
        f"{band}_{i//num_bands}"
        for i, (_, band) in enumerate(data.coords["time_band"].values)
    ]
    data = data.drop_vars(["time_band", "time", "band"])
    data.coords["time_band"] = new_bands_indices
    data = data.rename({"time_band": "band"}).transpose("band", "y", "x")
    return data
